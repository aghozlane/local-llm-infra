/**
 * Pure sizing formulas, verbatim from the validated plan (section
 * "Formules & constantes"). No React, no network: pure math only.
 *
 * Idealized tensor parallelism note: with G GPUs, each GPU holds 1/G of the
 * weights and KV and delivers 1/G of the bandwidth and FLOPS, so both
 * per-user decode throughput and prefill time are independent of G for a
 * given technology. The formulas below are therefore written per unit of
 * aggregate hardware bandwidth/FLOPS (the "FLOPS" argument of ttftSeconds
 * is the aggregate FLOP/s of the whole configuration, i.e. G x FLOPS_gpu).
 */
import type {
  GpuCountResult,
  GpuSpec,
  ModelSpec,
  Scenario,
  SizingResult,
} from './types';
import { bpwFor } from './quantization';
import { kvBytesPerToken, kvCacheBytes } from './kv';

// ---------------------------------------------------------------------------
// Unit constants
// ---------------------------------------------------------------------------

/**
 * 2^30 - one GiB. VRAM and RAM capacities are base-2 quantities (a card
 * marketed "24 GB" physically holds 24 GiB). Documented choice: the engine
 * expresses capacities in GiB and the French UI labels them "Go".
 */
export const BYTES_PER_GIB = 2 ** 30;

/** 10^9 - one decimal GB. GPU memory-bandwidth datasheet figures are SI. */
export const BYTES_PER_GB = 1e9;

/** Converts an exact byte count into GiB (base-2). */
export function bytesToGib(bytes: number): number {
  return bytes / BYTES_PER_GIB;
}

// ---------------------------------------------------------------------------
// Named constants (plan section "Formules & constantes") - calibrated against
// public llama.cpp benchmarks in a dedicated plan step (step 7).
// ---------------------------------------------------------------------------

/** Fixed VRAM overhead per sizing (plan: "overhead = 2 Go fixes"), in GiB. */
export const OVERHEAD_GIB = 2;

/** Achievable fraction of the catalog memory bandwidth (plan: BW_eff ~ 85 %). */
export const BW_EFFICIENCY = 0.85;

/** Prefill efficiency eta (plan: eta ~ 0.4). */
export const PREFILL_ETA = 0.4;

/** Default safety margin applied to the whole VRAM requirement (20 %). */
export const DEFAULT_SAFETY_MARGIN = 0.2;

/**
 * Fraction of a card's VRAM usable by the model (~93 %); the rest is taken
 * by the driver, display, fragmentation.
 */
export const USABLE_VRAM_FRACTION = 0.93;

/** v1 hardware-catalog bound: at most 8 GPUs ("v1 bound: <= 8 GPU"). */
export const MAX_GPUS_V1 = 8;

/** Minimum system RAM floor, in GiB (plan: "max(64 Go, ...)"). */
export const MIN_SYSTEM_RAM_GIB = 64;

/** System RAM scales as 1.5 x total VRAM. */
export const RAM_PER_VRAM_FACTOR = 1.5;

// ---------------------------------------------------------------------------
// Weights
// ---------------------------------------------------------------------------

/** Weights size in bytes: N x bpw / 8 (N = total parameters). */
export function weightBytes(totalParams: number, bpw: number): number {
  return (totalParams * bpw) / 8;
}

/**
 * Active parameters per token (A):
 * - MoE with identifiable experts: num_experts_per_tok x size_expert;
 * - dense, or MoE whose expert fields are missing: all parameters (N).
 *   (Design §7: an unidentifiable MoE is treated as dense.)
 */
export function activeParamsPerToken(spec: ModelSpec): number {
  if (
    spec.isMoe &&
    spec.numExpertsPerTok !== undefined &&
    spec.expertSize !== undefined
  ) {
    return spec.numExpertsPerTok * spec.expertSize;
  }
  return spec.totalParams;
}

// ---------------------------------------------------------------------------
// VRAM
// ---------------------------------------------------------------------------

/**
 * Required VRAM in GiB: (weights + KV + overhead) x (1 + safety margin).
 * The fixed overhead (2 GiB) is added ONCE and the margin applies to the
 * WHOLE sum ("then safety margin applied to the whole").
 *
 * @param wBytes weights size in bytes
 * @param kvBytes KV cache size in bytes
 * @param overheadGib fixed overhead in GiB (default OVERHEAD_GIB = 2)
 * @param safetyMargin margin as a fraction (default DEFAULT_SAFETY_MARGIN = 0.2)
 */
export function requiredVramGib(
  wBytes: number,
  kvBytes: number,
  overheadGib: number = OVERHEAD_GIB,
  safetyMargin: number = DEFAULT_SAFETY_MARGIN,
): number {
  return ((wBytes + kvBytes) / BYTES_PER_GIB + overheadGib) * (1 + safetyMargin);
}

/** VRAM usable by the model on one card: ~93 % of the card capacity. */
export function usableVramPerGpu(
  cardVramGib: number,
  usableFraction: number = USABLE_VRAM_FRACTION,
): number {
  return cardVramGib * usableFraction;
}

/** GPU count: ceil(required VRAM / usable VRAM per GPU), with the v1 cap signal. */
export function computeGpuCount(
  requiredVramGibValue: number,
  usableVramPerGpu: number,
): GpuCountResult {
  const rawCount = Math.ceil(requiredVramGibValue / usableVramPerGpu);
  return { gpuCount: rawCount, exceedsV1Cap: rawCount > MAX_GPUS_V1 };
}

// ---------------------------------------------------------------------------
// Decode throughput (roofline)
// ---------------------------------------------------------------------------

/** Average KV occupancy per sequence over a generation: inputTokens + outputTokens / 2. */
export function averageContextTokens(
  inputTokens: number,
  outputTokens: number,
): number {
  return inputTokens + outputTokens / 2;
}

/**
 * Decode throughput per user, in tokens/s:
 *
 *   tps_user = BW_eff / (A x bpw/8 + B x kv_bytes_per_token x average_context)
 *
 * where BW_eff = bandwidth x BW_EFFICIENCY (in bytes/s), B = simultaneous
 * sequences and A = active parameters per token. Verbatim from the plan.
 *
 * @param bwGbps      memory bandwidth in GB/s (decimal, as marketed)
 * @param activeParams active parameters per token (A)
 * @param bpw         effective bits per weight of the quantization
 * @param kvPerToken  KV bytes per token for one sequence
 * @param simultaneous simultaneous sequences (B)
 * @param avgContext  average context: inputTokens + outputTokens / 2
 * @param bwEfficiency fraction of bandwidth achievable (default BW_EFFICIENCY)
 */
export function decodeToksPerUser(
  bwGbps: number,
  activeParams: number,
  bpw: number,
  kvPerToken: number,
  simultaneous: number,
  avgContext: number,
  bwEfficiency: number = BW_EFFICIENCY,
): number {
  const bwEffBytesPerSec = bwGbps * BYTES_PER_GB * bwEfficiency;
  const bytesPerDecodeStepPerToken =
    activeParams * (bpw / 8) + simultaneous * kvPerToken * avgContext;
  return bwEffBytesPerSec / bytesPerDecodeStepPerToken;
}

/** Aggregate throughput in tokens/s: B x tps_user (continuous batching). */
export function aggregateToksPerSec(simultaneous: number, tpsPerUser: number): number {
  return simultaneous * tpsPerUser;
}

// ---------------------------------------------------------------------------
// TTFT (prefill)
// ---------------------------------------------------------------------------

/**
 * TTFT in seconds for one request: 2 x A x inputTokens / (FLOPS x eta).
 * eta = PREFILL_ETA ~ 0.4. Verbatim from the plan.
 *
 * @param flopsFp16Aggregate aggregate FP16 FLOP/s of the configuration
 *   (G x FLOPS_gpu under idealized tensor parallelism - see module docstring)
 * @param activeParams active parameters per token (A)
 * @param inputTokens prompt length in tokens
 * @param prefillEta prefill efficiency (default PREFILL_ETA)
 */
export function ttftSeconds(
  flopsFp16Aggregate: number,
  activeParams: number,
  inputTokens: number,
  prefillEta: number = PREFILL_ETA,
): number {
  return (2 * activeParams * inputTokens) / (flopsFp16Aggregate * prefillEta);
}

// ---------------------------------------------------------------------------
// Offered load vs capacity
// ---------------------------------------------------------------------------

/**
 * Offered load in tokens/s: totalUsers x reqPerUserPerMin x (in + out) / 60.
 * Verbatim from the plan ("offered load").
 */
export function offeredLoadToksPerSec(
  totalUsers: number,
  reqPerUserPerMin: number,
  inputTokens: number,
  outputTokens: number,
): number {
  return (totalUsers * reqPerUserPerMin * (inputTokens + outputTokens)) / 60;
}

/** True when the aggregate throughput absorbs the offered load (displayed check). */
export function supportsLoad(offeredLoad: number, aggregateToks: number): boolean {
  return offeredLoad <= aggregateToks;
}

// ---------------------------------------------------------------------------
// System RAM
// ---------------------------------------------------------------------------

/**
 * System RAM in GiB: max(64 GiB, 1.5 x total VRAM).
 *
 * @param vramTotalGib total VRAM in GiB. In SizingResult this is the required
 *   VRAM of the scenario (post-safety-margin); the recommendation step (plan
 *   step 5) may recompute it from the installed VRAM of a picked config.
 */
export function systemRamGib(vramTotalGib: number): number {
  return Math.max(MIN_SYSTEM_RAM_GIB, RAM_PER_VRAM_FACTOR * vramTotalGib);
}

// ---------------------------------------------------------------------------
// Edge cases and warnings (design §7)
// ---------------------------------------------------------------------------

/** Simultaneous sequences below 1 are clamped to 1. */
export function clampSimultaneous(value: number): number {
  return Math.max(1, value);
}

/** True when contextMax < inputTokens + outputTokens (inline warning). */
export function isContextTooSmall(scenario: Scenario): boolean {
  return scenario.contextMax < scenario.inputTokens + scenario.outputTokens;
}

// ---------------------------------------------------------------------------
// Orchestrator: one call producing the full SizingResult
// ---------------------------------------------------------------------------

/**
 * Computes the complete sizing for a model + scenario served on `gpu`
 * (idealized tensor parallelism - see module docstring).
 *
 * The per-user decode throughput is GPU-count independent; TTFT uses the
 * aggregate FLOPS of the computed GPU count.
 */
export function sizeForHardware(
  spec: ModelSpec,
  scenario: Scenario,
  gpu: GpuSpec,
): SizingResult {
  const simultaneous = clampSimultaneous(scenario.simultaneous);
  const bpw = bpwFor(scenario.quant);
  const weights = weightBytes(spec.totalParams, bpw);
  const kvPerToken = kvBytesPerToken(spec.numLayers, spec.numKvHeads, spec.headDim);
  const kv = kvCacheBytes(kvPerToken, scenario.contextMax, simultaneous);
  const margin = scenario.safetyMargin ?? DEFAULT_SAFETY_MARGIN;
  const vramGib = requiredVramGib(weights, kv, OVERHEAD_GIB, margin);
  const gpuNeeded = computeGpuCount(vramGib, usableVramPerGpu(gpu.vramGib));
  const active = activeParamsPerToken(spec);
  const avgContext = averageContextTokens(scenario.inputTokens, scenario.outputTokens);
  const tpsPerUser = decodeToksPerUser(
    gpu.bwGbps,
    active,
    bpw,
    kvPerToken,
    simultaneous,
    avgContext,
  );
  const aggregate = aggregateToksPerSec(simultaneous, tpsPerUser);
  const ttft = ttftSeconds(gpuNeeded.gpuCount * gpu.flopsFp16, active, scenario.inputTokens);
  const offeredLoad = offeredLoadToksPerSec(
    scenario.totalUsers,
    scenario.reqPerUserPerMin,
    scenario.inputTokens,
    scenario.outputTokens,
  );
  return {
    weightBytes: weights,
    kvBytes: kv,
    kvBytesPerToken: kvPerToken,
    weightGib: bytesToGib(weights),
    kvGib: bytesToGib(kv),
    overheadGib: OVERHEAD_GIB,
    safetyMargin: margin,
    requiredVramGib: vramGib,
    gpuCount: gpuNeeded.gpuCount,
    exceedsV1GpuCap: gpuNeeded.exceedsV1Cap,
    activeParams: active,
    averageContextTokens: avgContext,
    tpsPerUser,
    aggregateToksPerSec: aggregate,
    ttftSeconds: ttft,
    offeredLoadToksPerSec: offeredLoad,
    loadSupported: supportsLoad(offeredLoad, aggregate),
    ttftTargetMet:
      scenario.ttftTargetSec === undefined ? null : ttft <= scenario.ttftTargetSec,
    tpsTargetMet:
      scenario.minTpsPerUser === undefined ? null : tpsPerUser >= scenario.minTpsPerUser,
    systemRamGib: systemRamGib(vramGib),
    contextTooSmall: isContextTooSmall(scenario),
  };
}
