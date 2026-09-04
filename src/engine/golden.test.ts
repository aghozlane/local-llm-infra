import { describe, expect, it } from 'vitest';
import { sizeForHardware } from './index';
import type { GpuSpec, ModelSpec, Scenario } from './index';

/**
 * Golden scenario (plan section "Validation du golden scenario"):
 * Qwen/Qwen3-30B-A3B, Q4_K_M, 10 simultaneous / 30 total users, 3 req/min/user,
 * 2000 input + 500 output tokens, 16 K context. Targets: TTFT < 2 s,
 * >= 20 tok/s per user.
 *
 * The plan's hand-check figures ("~16 Go de KV", "~18 Go de poids") are
 * DECIMAL GB; this engine expresses capacities in GiB (base-2), so the
 * equivalents are: KV = 16,106,127,360 bytes = exactly 15 GiB ("~16 Go"),
 * weights = 18e9 bytes = ~16.76 GiB ("~18 Go"). Same order of magnitude.
 */
const GOLDEN_SPEC: ModelSpec = {
  totalParams: 30_000_000_000,
  numLayers: 48,
  numKvHeads: 4,
  headDim: 128,
  isMoe: true,
  numExpertsPerTok: 8,
  // Encodes the resolved active-parameter product: 8 experts x 412.5M = 3.3B
  // active params/token (plan: "~3 Md actifs par token" for Qwen3-30B-A3B).
  expertSize: 412_500_000,
};

const GOLDEN_SCENARIO: Scenario = {
  quant: 'Q4_K_M',
  simultaneous: 10,
  totalUsers: 30,
  reqPerUserPerMin: 3,
  inputTokens: 2000,
  outputTokens: 500,
  contextMax: 16_384,
  ttftTargetSec: 2,
  minTpsPerUser: 20,
};

/** Test fixture standing in for a catalog GPU (32 GiB, ~1.8 TB/s, ~100 TFLOPS FP16). */
const GOLDEN_GPU: GpuSpec = {
  name: 'GPU test 32 GiB (RTX 5090-class)',
  vramGib: 32,
  bwGbps: 1792,
  flopsFp16: 100e12,
};

const GOLDEN = sizeForHardware(GOLDEN_SPEC, GOLDEN_SCENARIO, GOLDEN_GPU);

describe('golden scenario: Qwen/Qwen3-30B-A3B Q4_K_M, 10 simult / 30 users', () => {
  it('KV per token matches the plan hand-check: 98,304 bytes = 96 KiB', () => {
    // 2 (K+V) x L(48) x kv_h(4) x d_h(128) x 2 (FP16)
    expect(GOLDEN.kvBytesPerToken).toBe(98_304);
    expect(GOLDEN.kvBytesPerToken).toBe(96 * 1024);
  });

  it('KV cache totals ~16 Go (plan decimal) = exactly 15 GiB for 16K x 10 sequences', () => {
    expect(GOLDEN.kvBytes).toBe(98_304 * 16_384 * 10);
    expect(GOLDEN.kvBytes).toBe(16_106_127_360);
    expect(GOLDEN.kvBytes).toBe(15 * 2 ** 30);
    expect(GOLDEN.kvGib).toBe(15);
    // Plan order of magnitude: "~16 Go de KV".
    expect(GOLDEN.kvGib).toBeGreaterThan(10);
    expect(GOLDEN.kvGib).toBeLessThan(20);
  });

  it('weights total ~18 Go (plan decimal) in Q4_K_M: 30e9 x 4.8 / 8 bytes', () => {
    expect(GOLDEN.weightBytes).toBe(18_000_000_000);
    expect(GOLDEN.weightGib).toBeCloseTo(16.7638, 3);
    // Plan order of magnitude: "~18 Go de poids".
    expect(GOLDEN.weightGib).toBeGreaterThan(10);
    expect(GOLDEN.weightGib).toBeLessThan(20);
  });

  it('KV is of the same order as weights and is the fast-growing term', () => {
    // Plan hand-check: "~16 Go de KV" against "~18 Go de poids" at 16K/10 seq:
    // same order of magnitude, weights slightly ahead AT THIS POINT. The plan's
    // "why KV dominates" argument is that KV grows with BOTH context and
    // simultaneity (weights do not), which the next assertions cover.
    expect(GOLDEN.kvGib).toBeGreaterThan(0.5 * GOLDEN.weightGib);
    expect(GOLDEN.kvGib).toBeLessThan(2 * GOLDEN.weightGib);
    const doubled = sizeForHardware(
      GOLDEN_SPEC,
      { ...GOLDEN_SCENARIO, contextMax: 32_768 },
      GOLDEN_GPU,
    );
    expect(doubled.kvBytes).toBe(2 * GOLDEN.kvBytes);
    expect(doubled.weightBytes).toBe(GOLDEN.weightBytes);
  });

  it('requires ~40.5 GiB of VRAM with the default 20 % margin', () => {
    // (16.7638 + 15 + 2) x 1.2 = 40.5166 GiB
    expect(GOLDEN.requiredVramGib).toBeCloseTo(40.5166, 3);
    expect(GOLDEN.safetyMargin).toBe(0.2);
    expect(GOLDEN.overheadGib).toBe(2);
    expect(GOLDEN.requiredVramGib).toBeGreaterThan(30);
    expect(GOLDEN.requiredVramGib).toBeLessThan(50);
  });

  it('needs 2 GPUs of 32 GiB (usable ~29.76 GiB each), within the v1 cap', () => {
    expect(GOLDEN.gpuCount).toBe(2);
    expect(GOLDEN.exceedsV1GpuCap).toBe(false);
  });

  it('uses 3.3B active parameters per token (MoE) and an average context of 2250', () => {
    expect(GOLDEN.activeParams).toBe(3_300_000_000);
    expect(GOLDEN.averageContextTokens).toBe(2000 + 500 / 2);
    expect(GOLDEN.averageContextTokens).toBe(2250);
  });

  it('decode throughput: ~363 tok/s per user, ~3634 tok/s aggregate', () => {
    // BW_eff = 1792 GB/s x 0.85 = 1.5232e12 B/s;
    // denominator = 3.3e9 x 0.6 + 10 x 98304 x 2250 = 4,191,840,000 bytes.
    expect(GOLDEN.tpsPerUser).toBeCloseTo(363.37, 1);
    expect(GOLDEN.tpsPerUser).toBeGreaterThan(300);
    expect(GOLDEN.tpsPerUser).toBeLessThan(400);
    expect(GOLDEN.aggregateToksPerSec).toBeCloseTo(3633.73, 1);
  });

  it('meets the golden targets: TTFT < 2 s and >= 20 tok/s per user', () => {
    // TTFT = 2 x 3.3e9 x 2000 / (2 GPUs x 100e12 FLOPS x 0.4) = 0.165 s.
    expect(GOLDEN.ttftSeconds).toBeCloseTo(0.165, 5);
    expect(GOLDEN.ttftTargetMet).toBe(true);
    expect(GOLDEN.tpsTargetMet).toBe(true);
  });

  it('offered load is 3750 tok/s and marginally exceeds the aggregate capacity', () => {
    // 30 users x 3 req/min x 2500 tokens / 60 = 3750 tok/s, vs ~3633.7 tok/s
    // sustainable: a documented near-miss that demonstrates the load check.
    // The plan's golden criteria only cover TTFT < 2 s and >= 20 tok/s/user,
    // both met; the offered-load verdict is reported as-is.
    expect(GOLDEN.offeredLoadToksPerSec).toBe(3750);
    expect(GOLDEN.loadSupported).toBe(false);
  });

  it('system RAM hits the 64 GiB floor (1.5 x 40.5 GiB < 64)', () => {
    expect(GOLDEN.systemRamGib).toBe(64);
  });

  it('raises no inline warning for the golden context', () => {
    expect(GOLDEN.contextTooSmall).toBe(false);
  });
});
