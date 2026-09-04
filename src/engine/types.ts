/**
 * Pure calculation engine types (plan step 2).
 *
 * The engine has ZERO dependency on React, the DOM or the network: it maps
 * typed inputs to typed outputs through pure functions.
 *
 * Unit conventions (IMPORTANT - read before using any numeric field):
 * - Memory CAPACITIES (VRAM, RAM) are expressed in GiB (base-2, 2^30 bytes).
 *   GPU VRAM is physically organized in binary multiples (a card marketed
 *   "24 GB" physically holds 24 GiB), so GiB is the physically accurate unit;
 *   the French UI labels these figures "Go". See BYTES_PER_GIB in formulas.ts.
 * - Memory BANDWIDTH figures are decimal (GB/s = 10^9 bytes/s), as marketed
 *   in GPU datasheets. See BYTES_PER_GB in formulas.ts.
 * - Byte-level intermediates (weights, KV cache) are exact byte counts.
 */

/** Supported quantization formats, keyed by their GGUF-style name. */
export type QuantName = 'FP16' | 'Q8_0' | 'Q6_K' | 'Q5_K_M' | 'Q4_K_M';

/**
 * Model description, resolved from Hugging Face (plan step 3 owns the
 * resolution; the engine only consumes it).
 */
export interface ModelSpec {
  /**
   * Total parameter count, ALL experts included for MoE (they are all
   * resident in memory), e.g. 30e9 for a 30B model.
   */
  readonly totalParams: number;
  /** Transformer layer count (config `num_hidden_layers`). */
  readonly numLayers: number;
  /** KV heads per layer (config `num_key_value_heads` or fallback). */
  readonly numKvHeads: number;
  /** Head dimension (config `head_dim` or `hidden_size / num_attention_heads`). */
  readonly headDim: number;
  /** True when the model is a Mixture-of-Experts. */
  readonly isMoe: boolean;
  /** MoE only: experts routed per token (config `num_experts_per_tok`). */
  readonly numExpertsPerTok?: number;
  /** MoE only: parameter count of one expert (includes its MLP share). */
  readonly expertSize?: number;
}

/** Load profile and targets of the sizing question. */
export interface Scenario {
  /** Quantization applied to the weights. */
  readonly quant: QuantName;
  /**
   * Number of simultaneous sequences (B, "sequences simultaneously").
   * Values below 1 are clamped to 1 (design §7 edge case) - see
   * clampSimultaneous in formulas.ts.
   */
  readonly simultaneous: number;
  /** Total distinct users over the workload (for the offered-load check). */
  readonly totalUsers: number;
  /** Requests per user per minute. */
  readonly reqPerUserPerMin: number;
  /** Average input (prompt) tokens per request. */
  readonly inputTokens: number;
  /** Average output tokens per request. */
  readonly outputTokens: number;
  /** Maximum context length kept in KV cache ("max context"). */
  readonly contextMax: number;
  /**
   * Safety margin applied to the whole VRAM requirement, as a FRACTION
   * (0.2 = 20 %). Default: DEFAULT_SAFETY_MARGIN (0.2).
   */
  readonly safetyMargin?: number;
  /** Maximum acceptable TTFT in seconds. Absent = no TTFT target. */
  readonly ttftTargetSec?: number;
  /** Minimum acceptable tokens/s per user. Absent = no throughput target. */
  readonly minTpsPerUser?: number;
}

/**
 * Minimal hardware characteristics consumed by the engine formulas.
 * The hardware catalog (plan step 4) extends or wraps this interface with
 * price, TDP, etc.; the engine only needs the perf/memory fields.
 */
export interface GpuSpec {
  readonly name: string;
  /** Marketed VRAM capacity, in GiB (base-2). */
  readonly vramGib: number;
  /** Memory bandwidth, in GB/s (decimal, as marketed). */
  readonly bwGbps: number;
  /** FP16 (dense) compute throughput, in FLOP/s. */
  readonly flopsFp16: number;
}

/** Result of the GPU-count computation, including the v1 cap signal. */
export interface GpuCountResult {
  /**
   * Raw ceil-based GPU count, NOT clamped to the cap: an impossible
   * requirement must stay visible instead of silently saturating at 8.
   */
  readonly gpuCount: number;
  /** True when the count exceeds MAX_GPUS_V1 ("out of v1 catalog"). */
  readonly exceedsV1Cap: boolean;
}

/** Full sizing output for one model + scenario + GPU configuration. */
export interface SizingResult {
  // --- Memory (exact bytes + GiB figures for display) ---
  /** Weights size in bytes: N x bpw / 8. */
  readonly weightBytes: number;
  /** Total KV cache in bytes: per-token footprint x contextMax x simultaneous. */
  readonly kvBytes: number;
  /** KV bytes per token for ONE sequence. */
  readonly kvBytesPerToken: number;
  /** Weights size in GiB (base-2). */
  readonly weightGib: number;
  /** KV cache size in GiB (base-2). */
  readonly kvGib: number;
  /** Fixed VRAM overhead, in GiB (plan: "overhead = 2 GB fixed"). */
  readonly overheadGib: number;
  /** Safety margin actually applied, as a fraction (0.2 = 20 %). */
  readonly safetyMargin: number;
  /**
   * Required VRAM in GiB: (weights + KV + overhead) x (1 + safety margin) -
   * the margin applies to the WHOLE (plan: "applied to the whole").
   */
  readonly requiredVramGib: number;
  /** Number of GPUs needed: ceil(required VRAM / usable VRAM per GPU). */
  readonly gpuCount: number;
  /** True when gpuCount exceeds the v1 cap (MAX_GPUS_V1). */
  readonly exceedsV1GpuCap: boolean;

  // --- Performance ---
  /** Active parameters per token (A): MoE num_experts_per_tok x size_expert, else N. */
  readonly activeParams: number;
  /** Average KV occupancy per sequence: inputTokens + outputTokens / 2. */
  readonly averageContextTokens: number;
  /** Decode throughput per user, in tokens/s (roofline). */
  readonly tpsPerUser: number;
  /** Aggregate throughput in tokens/s: simultaneous x tpsPerUser (continuous batching). */
  readonly aggregateToksPerSec: number;
  /** TTFT in seconds for the average input prompt: 2 x A x inputTokens / (FLOPS x eta). */
  readonly ttftSeconds: number;

  // --- Offered load vs capacity ---
  /** Offered load in tokens/s: totalUsers x reqPerUserPerMin x (in + out) / 60. */
  readonly offeredLoadToksPerSec: number;
  /** True when the offered load is absorbed by the aggregate throughput. */
  readonly loadSupported: boolean;

  // --- Target checks (null when the corresponding target was not provided) ---
  /** True when TTFT meets the ttftTargetSec target. */
  readonly ttftTargetMet: boolean | null;
  /** True when per-user throughput meets the minTpsPerUser target. */
  readonly tpsTargetMet: boolean | null;

  // --- System RAM ---
  /** System RAM in GiB: max(64 GiB, 1.5 x requiredVramGib). */
  readonly systemRamGib: number;

  // --- Inline warnings (design §7) ---
  /** True when contextMax < inputTokens + outputTokens. */
  readonly contextTooSmall: boolean;
}
