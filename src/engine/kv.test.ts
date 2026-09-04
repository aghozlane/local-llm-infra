import { describe, expect, it } from 'vitest';
import {
  KV_BYTES_PER_ELEMENT,
  KV_PROJECTIONS,
  kvBytesPerToken,
  kvCacheBytes,
} from './index';

describe('KV cache formulas', () => {
  it('stores K and V in FP16 (2 bytes per element)', () => {
    expect(KV_PROJECTIONS).toBe(2);
    expect(KV_BYTES_PER_ELEMENT).toBe(2);
  });

  it('computes KV bytes per token for a dense multi-head model (kv_h = heads)', () => {
    // L=32, kv_h=32, d_h=128 -> 2*32*32*128*2 = 524,288 bytes/token/sequence
    expect(kvBytesPerToken(32, 32, 128)).toBe(524_288);
  });

  it('computes KV bytes per token for a GQA model (fewer KV heads than heads)', () => {
    // L=32, kv_h=8, d_h=128 -> 2*32*8*128*2 = 131,072 bytes/token/sequence
    expect(kvBytesPerToken(32, 8, 128)).toBe(131_072);
  });

  it('matches the golden model geometry: Qwen3-30B-A3B (L=48, kv_h=4, d_h=128)', () => {
    // 2*48*4*128*2 = 98,304 bytes = 96 KiB per token and per sequence
    expect(kvBytesPerToken(48, 4, 128)).toBe(98_304);
    expect(kvBytesPerToken(48, 4, 128)).toBe(96 * 1024);
  });

  it('scales the total KV cache linearly with context and simultaneous sequences', () => {
    const perToken = kvBytesPerToken(48, 4, 128);
    // Golden check: 16 K context x 10 sequences = exactly 15 GiB (2^30 bytes)
    expect(kvCacheBytes(perToken, 16_384, 10)).toBe(16_106_127_360);
    expect(kvCacheBytes(perToken, 16_384, 10)).toBe(15 * 2 ** 30);
    expect(kvCacheBytes(perToken, 16_384, 20)).toBe(
      2 * kvCacheBytes(perToken, 16_384, 10),
    );
    expect(kvCacheBytes(perToken, 32_768, 10)).toBe(
      2 * kvCacheBytes(perToken, 16_384, 10),
    );
  });

  it('is invariant to parameter count: KV depends only on L, kv_h, d_h', () => {
    // A MoE with the same attention geometry has the SAME KV footprint;
    // MoE only changes the active parameters per token (A).
    expect(kvBytesPerToken(48, 4, 128)).toBe(kvBytesPerToken(48, 4, 128));
    expect(kvCacheBytes(98_304, 16_384, 10)).toBe(16_106_127_360);
  });
});
