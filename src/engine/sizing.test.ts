import { describe, expect, it } from 'vitest';
import {
  activeParamsPerToken,
  aggregateToksPerSec,
  bpwFor,
  bytesToGib,
  clampSimultaneous,
  computeGpuCount,
  decodeToksPerUser,
  kvBytesPerToken,
  kvCacheBytes,
  offeredLoadToksPerSec,
  requiredVramGib,
  sizeForHardware,
  supportsLoad,
  systemRamGib,
  ttftSeconds,
  usableVramPerGpu,
} from './index';
import type { GpuSpec, ModelSpec, Scenario } from './index';

/** Simple dense model: 8B parameters, L=32, kv_h=8, d_h=128 (GQA geometry). */
const SPEC: ModelSpec = {
  totalParams: 8_000_000_000,
  numLayers: 32,
  numKvHeads: 8,
  headDim: 128,
  isMoe: false,
};

const SCENARIO: Scenario = {
  quant: 'Q4_K_M',
  simultaneous: 4,
  totalUsers: 10,
  reqPerUserPerMin: 2,
  inputTokens: 500,
  outputTokens: 250,
  contextMax: 8_192,
};

const GPU: GpuSpec = {
  name: 'GPU test 24 GiB',
  vramGib: 24,
  bwGbps: 1008,
  flopsFp16: 100e12,
};

const RESULT = sizeForHardware(SPEC, SCENARIO, GPU);

describe('sizeForHardware orchestrator', () => {
  it('wires every field to the underlying formulas', () => {
    const bpw = bpwFor(SCENARIO.quant);
    const expectedKvPerToken = kvBytesPerToken(SPEC.numLayers, SPEC.numKvHeads, SPEC.headDim);
    const expectedWeights = (SPEC.totalParams * bpw) / 8;
    const expectedKv = kvCacheBytes(expectedKvPerToken, SCENARIO.contextMax, 4);
    const expectedVram = requiredVramGib(expectedWeights, expectedKv);
    const expectedActive = activeParamsPerToken(SPEC);
    const expectedAvgCtx = 500 + 250 / 2;
    const expectedTps = decodeToksPerUser(
      GPU.bwGbps,
      expectedActive,
      bpw,
      expectedKvPerToken,
      4,
      expectedAvgCtx,
    );
    const expectedCount = computeGpuCount(expectedVram, usableVramPerGpu(GPU.vramGib));
    const expectedTtft = ttftSeconds(
      expectedCount.gpuCount * GPU.flopsFp16,
      expectedActive,
      SCENARIO.inputTokens,
    );
    const expectedLoad = offeredLoadToksPerSec(
      SCENARIO.totalUsers,
      SCENARIO.reqPerUserPerMin,
      SCENARIO.inputTokens,
      SCENARIO.outputTokens,
    );

    expect(RESULT.kvBytesPerToken).toBe(expectedKvPerToken);
    expect(RESULT.weightBytes).toBe(expectedWeights);
    expect(RESULT.kvBytes).toBe(expectedKv);
    expect(RESULT.weightGib).toBeCloseTo(bytesToGib(expectedWeights), 9);
    expect(RESULT.kvGib).toBeCloseTo(bytesToGib(expectedKv), 9);
    expect(RESULT.requiredVramGib).toBeCloseTo(expectedVram, 9);
    expect(RESULT.gpuCount).toBe(expectedCount.gpuCount);
    expect(RESULT.exceedsV1GpuCap).toBe(expectedCount.exceedsV1Cap);
    expect(RESULT.activeParams).toBe(expectedActive);
    expect(RESULT.averageContextTokens).toBe(expectedAvgCtx);
    expect(RESULT.tpsPerUser).toBeCloseTo(expectedTps, 9);
    expect(RESULT.aggregateToksPerSec).toBeCloseTo(aggregateToksPerSec(4, expectedTps), 9);
    expect(RESULT.ttftSeconds).toBeCloseTo(expectedTtft, 9);
    expect(RESULT.offeredLoadToksPerSec).toBeCloseTo(expectedLoad, 9);
    expect(RESULT.loadSupported).toBe(supportsLoad(expectedLoad, 4 * expectedTps));
    expect(RESULT.systemRamGib).toBeCloseTo(systemRamGib(expectedVram), 9);
  });

  it('computes exact spot values for the fixture', () => {
    // KV per token: 2*32*8*128*2 = 131,072 bytes; x 8192 x 4 = exactly 4 GiB.
    expect(RESULT.kvBytesPerToken).toBe(131_072);
    expect(RESULT.kvBytes).toBe(4_294_967_296);
    // Weights: 8e9 x 4.8 / 8 = 4.8e9 bytes.
    expect(RESULT.weightBytes).toBeCloseTo(4_800_000_000, 6);
    // TTFT: 2 x 8e9 x 500 / (1 GPU x 100e12 x 0.4) = 0.2 s.
    expect(RESULT.ttftSeconds).toBeCloseTo(0.2, 9);
    // Offered load: 10 x 2 x 750 / 60 = 250 tok/s.
    expect(RESULT.offeredLoadToksPerSec).toBe(250);
    expect(RESULT.averageContextTokens).toBe(625);
    // Aggregate absorbs the offered load (4 x ~167 > 250).
    expect(RESULT.loadSupported).toBe(true);
  });

  it('fills the default safety margin and default overhead', () => {
    expect(RESULT.safetyMargin).toBe(0.2);
    expect(RESULT.overheadGib).toBe(2);
  });

  it('reports null target checks when no target is provided', () => {
    const noTargets = sizeForHardware(SPEC, { ...SCENARIO, ttftTargetSec: undefined, minTpsPerUser: undefined }, GPU);
    expect(noTargets.ttftTargetMet).toBeNull();
    expect(noTargets.tpsTargetMet).toBeNull();
  });

  it('checks targets inclusively against computed values', () => {
    // TTFT 0.2 s <= 0.5 s target -> met; tps ~167 < 200 target -> not met.
    const withTargets = sizeForHardware(
      SPEC,
      { ...SCENARIO, ttftTargetSec: 0.5, minTpsPerUser: 200 },
      GPU,
    );
    expect(withTargets.ttftTargetMet).toBe(true);
    expect(withTargets.tpsTargetMet).toBe(false);
    expect(withTargets.ttftSeconds).toBeCloseTo(0.2, 9);
    expect(withTargets.tpsPerUser).toBeLessThan(200);
  });

  it('respects the default safety margin override in the scenario', () => {
    const relaxed = sizeForHardware(SPEC, { ...SCENARIO, safetyMargin: 0.5 }, GPU);
    const expected = requiredVramGib(RESULT.weightBytes, RESULT.kvBytes, 2, 0.5);
    expect(relaxed.safetyMargin).toBe(0.5);
    expect(relaxed.requiredVramGib).toBeCloseTo(expected, 9);
    expect(relaxed.requiredVramGib).toBeGreaterThan(RESULT.requiredVramGib);
  });

  it('clamps simultaneous sequences below 1 to 1 (design §7)', () => {
    const clamped = sizeForHardware(SPEC, { ...SCENARIO, simultaneous: 0 }, GPU);
    const explicit = sizeForHardware(SPEC, { ...SCENARIO, simultaneous: 1 }, GPU);
    expect(clamped).toEqual(explicit);
    // Sanity: aggregate of the clamped run equals the per-user throughput.
    expect(clamped.aggregateToksPerSec).toBeCloseTo(clamped.tpsPerUser, 9);
  });

  it('signals the v1 cap when the requirement forces more than 8 GPUs', () => {
    const tiny = sizeForHardware(SPEC, SCENARIO, { ...GPU, name: 'GPU test 1 GiB', vramGib: 1 });
    const usable = usableVramPerGpu(1);
    const expectedCount = computeGpuCount(RESULT.requiredVramGib, usable);
    expect(tiny.gpuCount).toBe(expectedCount.gpuCount);
    expect(tiny.gpuCount).toBeGreaterThan(8);
    expect(tiny.exceedsV1GpuCap).toBe(true);
  });

  it('keeps per-user decode throughput GPU-count independent, TTFT scales with count', () => {
    // Same bandwidth, 4x smaller cards -> 4x more GPUs, same tps_user, 1/4 TTFT.
    const small = sizeForHardware(SPEC, SCENARIO, { ...GPU, name: 'GPU test 4 GiB', vramGib: 4 });
    expect(small.gpuCount).toBe(4 * RESULT.gpuCount);
    expect(small.tpsPerUser).toBe(RESULT.tpsPerUser);
    expect(small.ttftSeconds).toBeCloseTo(RESULT.ttftSeconds * RESULT.gpuCount / small.gpuCount, 9);
  });

  it('flags a context smaller than input + output tokens', () => {
    const small = sizeForHardware(SPEC, { ...SCENARIO, contextMax: 700 }, GPU);
    expect(small.contextTooSmall).toBe(true);
    const ok = sizeForHardware(SPEC, { ...SCENARIO, contextMax: 800 }, GPU);
    expect(ok.contextTooSmall).toBe(false);
  });

  it('is stable when the same inputs are recomputed', () => {
    expect(sizeForHardware(SPEC, SCENARIO, GPU)).toEqual(RESULT);
  });

  it('applies clampSimultaneous before every downstream computation', () => {
    expect(clampSimultaneous(-5)).toBe(1);
    const clamped = sizeForHardware(SPEC, { ...SCENARIO, simultaneous: -5 }, GPU);
    // B is clamped before the KV computation: 131,072 x 8,192 x 1 = exactly 2^30 bytes.
    expect(clamped.kvBytes).toBe(1_073_741_824);
    expect(clamped.kvBytes).toBe(2 ** 30);
    // With B=1, the aggregate equals the per-user throughput.
    expect(clamped.aggregateToksPerSec).toBeCloseTo(clamped.tpsPerUser, 9);
  });
});
