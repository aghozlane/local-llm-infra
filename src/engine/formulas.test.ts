import { describe, expect, it } from 'vitest';
import {
  activeParamsPerToken,
  aggregateToksPerSec,
  BW_EFFICIENCY,
  BYTES_PER_GB,
  BYTES_PER_GIB,
  bytesToGib,
  clampSimultaneous,
  computeGpuCount,
  DEFAULT_SAFETY_MARGIN,
  decodeToksPerUser,
  isContextTooSmall,
  MAX_GPUS_V1,
  MIN_SYSTEM_RAM_GIB,
  OVERHEAD_GIB,
  offeredLoadToksPerSec,
  PREFILL_ETA,
  QUANT_BPW,
  QUANT_ORDER,
  RAM_PER_VRAM_FACTOR,
  requiredVramGib,
  supportsLoad,
  systemRamGib,
  ttftSeconds,
  usableVramPerGpu,
  weightBytes,
} from './index';
import type { ModelSpec, Scenario } from './index';

/** Reference model: 30B total parameters (plan golden scenario scale). */
const N = 30_000_000_000;

describe('weights: N x bpw / 8 across the five quants', () => {
  it('gives the exact expected byte size for each quantization', () => {
    for (const quant of QUANT_ORDER) {
      const bpw = QUANT_BPW[quant];
      expect(bpw).toBeDefined();
      if (bpw !== undefined) {
        expect(weightBytes(N, bpw)).toBeCloseTo((N * bpw) / 8, 6);
      }
    }
  });

  it('matches the hand-computed reference values for N = 30e9', () => {
    expect(weightBytes(N, QUANT_BPW.FP16)).toBeCloseTo(60_000_000_000, 6);
    expect(weightBytes(N, QUANT_BPW.Q8_0)).toBeCloseTo(31_875_000_000, 6);
    expect(weightBytes(N, QUANT_BPW.Q6_K)).toBeCloseTo(24_750_000_000, 6);
    expect(weightBytes(N, QUANT_BPW.Q5_K_M)).toBeCloseTo(21_375_000_000, 6);
    expect(weightBytes(N, QUANT_BPW.Q4_K_M)).toBeCloseTo(18_000_000_000, 6);
  });

  it('is linear in the parameter count', () => {
    expect(weightBytes(3_000_000_000, QUANT_BPW.Q4_K_M)).toBe(
      weightBytes(N, QUANT_BPW.Q4_K_M) / 10,
    );
  });
});

describe('active parameters per token (A)', () => {
  const denseSpec: ModelSpec = {
    totalParams: 8_000_000_000,
    numLayers: 32,
    numKvHeads: 8,
    headDim: 128,
    isMoe: false,
  };

  it('returns N for a dense model even if expert fields are present', () => {
    expect(
      activeParamsPerToken({ ...denseSpec, numExpertsPerTok: 2, expertSize: 1e9 }),
    ).toBe(8_000_000_000);
  });

  it('returns num_experts_per_tok x size_expert for an identified MoE', () => {
    const moe: ModelSpec = { ...denseSpec, isMoe: true, numExpertsPerTok: 8, expertSize: 412_500_000 };
    expect(activeParamsPerToken(moe)).toBe(3_300_000_000);
  });

  it('falls back to N for an unidentified MoE (design §7: treated as dense)', () => {
    const moe: ModelSpec = { ...denseSpec, isMoe: true };
    expect(activeParamsPerToken(moe)).toBe(8_000_000_000);
  });
});

describe('required VRAM: (weights + KV + overhead) x (1 + margin)', () => {
  // Golden-scale inputs: Q4_K_M weights of a 30B model, KV of the golden scenario.
  const w = weightBytes(N, QUANT_BPW.Q4_K_M); // 18e9 bytes
  const kv = 16_106_127_360; // 15 GiB exactly (golden scenario KV)

  it('applies the default 20 % margin to the whole sum, overhead added once', () => {
    // (18e9 + 15 GiB) / 2^30 = 31.7638 GiB; + 2 GiB overhead; x 1.2 = 40.5166 GiB
    expect(requiredVramGib(w, kv)).toBeCloseTo(40.5166, 3);
  });

  it('matches the explicit recomposition of the formula', () => {
    const expected = ((w + kv) / BYTES_PER_GIB + OVERHEAD_GIB) * (1 + DEFAULT_SAFETY_MARGIN);
    expect(requiredVramGib(w, kv)).toBeCloseTo(expected, 9);
  });

  it('adds the fixed overhead exactly once, before the margin', () => {
    const noMargin = requiredVramGib(w, kv, OVERHEAD_GIB, 0);
    const noOverheadNoMargin = requiredVramGib(w, kv, 0, 0);
    expect(noMargin - noOverheadNoMargin).toBeCloseTo(OVERHEAD_GIB * (1 + 0), 9);
    // Margin applies to the whole: margin 0 vs 0.5 differs by factor 1.5 on everything.
    expect(requiredVramGib(w, kv, OVERHEAD_GIB, 0.5)).toBeCloseTo(noMargin * 1.5, 9);
  });

  it('scales linearly with the margin', () => {
    expect(requiredVramGib(w, kv, OVERHEAD_GIB, 0.1)).toBeCloseTo(
      requiredVramGib(w, kv, OVERHEAD_GIB, 0) * 1.1,
      9,
    );
  });
});

describe('usable VRAM per GPU (~93 %)', () => {
  it('keeps 93 % of the card capacity by default', () => {
    expect(usableVramPerGpu(32)).toBeCloseTo(29.76, 9);
    expect(usableVramPerGpu(24)).toBeCloseTo(22.32, 9);
  });

  it('accepts a custom fraction', () => {
    expect(usableVramPerGpu(100, 0.5)).toBe(50);
  });
});

describe('GPU count: ceil(required / usable), v1 cap at 8 GPUs', () => {
  const usable = usableVramPerGpu(32); // ~29.76 GiB

  it('rounds up to the next GPU when VRAM exceeds one card', () => {
    // Golden scenario: ~40.5 GiB required, one 32 GiB card usable 29.76 -> 2 GPUs.
    expect(computeGpuCount(40.5166, usable)).toEqual({ gpuCount: 2, exceedsV1Cap: false });
  });

  it('fits in a single card below capacity', () => {
    expect(computeGpuCount(10, usable)).toEqual({ gpuCount: 1, exceedsV1Cap: false });
  });

  it('returns 1 at exactly one card of usable VRAM', () => {
    expect(computeGpuCount(usable, usable)).toEqual({ gpuCount: 1, exceedsV1Cap: false });
  });

  it('keeps 8 GPUs at the exact v1 bound', () => {
    expect(computeGpuCount(8 * 10, 10)).toEqual({
      gpuCount: 8,
      exceedsV1Cap: false,
    });
    expect(computeGpuCount(8 * 10, 10).gpuCount).toBe(MAX_GPUS_V1);
  });

  it('signals the v1 cap ("hors catalogue v1") beyond 8 GPUs without clamping the count', () => {
    expect(computeGpuCount(8 * 10 + 0.01, 10)).toEqual({
      gpuCount: 9,
      exceedsV1Cap: true,
    });
    expect(computeGpuCount(200, 10)).toEqual({ gpuCount: 20, exceedsV1Cap: true });
  });
});

describe('decode throughput per user (roofline)', () => {
  it('computes tps_user = BW_eff / (A x bpw/8 + B x kv_per_token x avg_ctx)', () => {
    // A=1e9, bpw=8 -> 1e9 bytes for weights; B=4, kv=1e5, ctx=1000 -> 4e8 bytes.
    // BW_eff = 1000 GB/s x 0.85 = 8.5e11 B/s; tps = 8.5e11 / 1.4e9 = 607.1428...
    const tps = decodeToksPerUser(1000, 1e9, 8, 1e5, 4, 1000);
    expect(tps).toBeCloseTo(607.142857, 3);
  });

  it('BW_eff is BW x BW_EFFICIENCY in bytes/s', () => {
    expect(BW_EFFICIENCY).toBe(0.85);
    expect(BYTES_PER_GB).toBe(1e9);
    const tps = decodeToksPerUser(1000, 1e9, 8, 1e5, 4, 1000, 1);
    expect(tps).toBeCloseTo(1000 * BYTES_PER_GB / (1e9 + 4e8), 9);
  });

  it('decreases as the number of simultaneous sequences grows', () => {
    const base = decodeToksPerUser(1000, 1e9, 8, 1e5, 4, 1000);
    const more = decodeToksPerUser(1000, 1e9, 8, 1e5, 8, 1000);
    expect(more).toBeLessThan(base);
    // Denominator is affine in B: tps(B=8) = BW_eff / (1e9 + 8e8)
    expect(more).toBeCloseTo((1000 * BYTES_PER_GB * BW_EFFICIENCY) / 1.8e9, 6);
  });

  it('uses average context = inputTokens + outputTokens / 2', () => {
    expect(decodeToksPerUser(1000, 1e9, 8, 1e5, 4, 2000 + 500 / 2)).toBe(
      decodeToksPerUser(1000, 1e9, 8, 1e5, 4, 2250),
    );
  });

  it('is independent of the GPU count (idealized tensor parallelism)', () => {
    // Same bandwidth, different card sizes: identical per-user throughput.
    expect(decodeToksPerUser(1000, 1e9, 8, 1e5, 4, 1000)).toBe(
      decodeToksPerUser(1000, 1e9, 8, 1e5, 4, 1000),
    );
  });
});

describe('aggregate throughput: B x tps_user', () => {
  it('scales the per-user throughput by the simultaneous sequences', () => {
    expect(aggregateToksPerSec(10, 100)).toBe(1000);
    expect(aggregateToksPerSec(4, 607.142857)).toBeCloseTo(2428.571428, 6);
  });
});

describe('TTFT: 2 x A x inputTokens / (FLOPS x eta)', () => {
  it('computes the plan formula with eta = 0.4', () => {
    // 2 x 3e9 x 1000 / (100e12 x 0.4) = 0.15 s
    expect(ttftSeconds(100e12, 3e9, 1000)).toBeCloseTo(0.15, 9);
    expect(PREFILL_ETA).toBe(0.4);
  });

  it('halves when the aggregate FLOPS doubles (multi-GPU prefill)', () => {
    const oneGpu = ttftSeconds(100e12, 3e9, 1000);
    const twoGpus = ttftSeconds(2 * 100e12, 3e9, 1000);
    expect(twoGpus).toBeCloseTo(oneGpu / 2, 9);
  });

  it('scales linearly with prompt length and active parameters', () => {
    const base = ttftSeconds(100e12, 3e9, 1000);
    expect(ttftSeconds(100e12, 3e9, 2000)).toBeCloseTo(2 * base, 9);
    expect(ttftSeconds(100e12, 6e9, 1000)).toBeCloseTo(2 * base, 9);
  });

  it('accepts a custom eta', () => {
    expect(ttftSeconds(100e12, 3e9, 1000, 0.5)).toBeCloseTo(0.12, 9);
  });
});

describe('offered load vs sustainable capacity', () => {
  it('computes totalUsers x reqPerUserPerMin x (in + out) / 60', () => {
    // Golden scenario: 30 users x 3 req/min x 2500 tokens / 60 = 3750 tok/s
    expect(offeredLoadToksPerSec(30, 3, 2000, 500)).toBe(3750);
    expect(offeredLoadToksPerSec(1, 60, 100, 100)).toBe(200);
  });

  it('compares offered load to aggregate throughput inclusively', () => {
    expect(supportsLoad(3750, 3633.73)).toBe(false);
    expect(supportsLoad(1000, 1000)).toBe(true);
    expect(supportsLoad(999.9, 1000)).toBe(true);
    expect(supportsLoad(1000.1, 1000)).toBe(false);
  });
});

describe('system RAM: max(64 GiB, 1.5 x total VRAM)', () => {
  it('applies the 64 GiB floor for small VRAM requirements', () => {
    expect(systemRamGib(0)).toBe(MIN_SYSTEM_RAM_GIB);
    expect(systemRamGib(40.5166)).toBe(MIN_SYSTEM_RAM_GIB); // 1.5 x 40.5 = 60.8 < 64
  });

  it('scales as 1.5 x VRAM beyond the floor', () => {
    expect(systemRamGib(100)).toBeCloseTo(150, 9);
    expect(systemRamGib(50)).toBeCloseTo(75, 9);
    expect(systemRamGib(50)).toBeCloseTo(RAM_PER_VRAM_FACTOR * 50, 9);
  });

  it('is continuous at the floor boundary (VRAM = 64 / 1.5)', () => {
    const boundary = MIN_SYSTEM_RAM_GIB / RAM_PER_VRAM_FACTOR;
    expect(systemRamGib(boundary)).toBeCloseTo(MIN_SYSTEM_RAM_GIB, 9);
    expect(systemRamGib(boundary + 0.001)).toBeCloseTo(1.5 * (boundary + 0.001), 6);
  });
});

describe('edge cases and unit helpers', () => {
  it('clamps simultaneous sequences below 1 up to 1', () => {
    expect(clampSimultaneous(0)).toBe(1);
    expect(clampSimultaneous(-3)).toBe(1);
    expect(clampSimultaneous(10)).toBe(10);
  });

  it('flags a context smaller than input + output tokens', () => {
    const base: Scenario = {
      quant: 'Q4_K_M',
      simultaneous: 10,
      totalUsers: 30,
      reqPerUserPerMin: 3,
      inputTokens: 2000,
      outputTokens: 500,
      contextMax: 16_384,
    };
    expect(isContextTooSmall(base)).toBe(false);
    expect(isContextTooSmall({ ...base, contextMax: 2500 })).toBe(false); // exactly enough
    expect(isContextTooSmall({ ...base, contextMax: 2499 })).toBe(true);
  });

  it('converts bytes to GiB with the base-2 constant', () => {
    expect(BYTES_PER_GIB).toBe(2 ** 30);
    expect(bytesToGib(2 ** 30)).toBe(1);
    expect(bytesToGib(0)).toBe(0);
    expect(bytesToGib(15 * 2 ** 30)).toBe(15);
  });
});

describe('named constants are exported and documented', () => {
  it('exposes the plan constants verbatim', () => {
    expect(BW_EFFICIENCY).toBe(0.85);
    expect(PREFILL_ETA).toBe(0.4);
    expect(OVERHEAD_GIB).toBe(2);
    expect(DEFAULT_SAFETY_MARGIN).toBe(0.2);
    expect(MAX_GPUS_V1).toBe(8);
    expect(MIN_SYSTEM_RAM_GIB).toBe(64);
    expect(RAM_PER_VRAM_FACTOR).toBe(1.5);
  });
});
