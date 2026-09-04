import { describe, expect, it } from 'vitest';
import { QUANT_BPW, QUANT_ORDER, bpwFor } from './index';

describe('quantization table (plan-verbatim values, must not be derived)', () => {
  it('exposes exactly the five supported quantizations in canonical order', () => {
    expect(QUANT_ORDER).toEqual(['FP16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M']);
  });

  it('maps each quantization to its exact effective bits per weight', () => {
    expect(QUANT_BPW.FP16).toBe(16);
    expect(QUANT_BPW.Q8_0).toBe(8.5);
    expect(QUANT_BPW.Q6_K).toBe(6.6);
    expect(QUANT_BPW.Q5_K_M).toBe(5.7);
    expect(QUANT_BPW.Q4_K_M).toBe(4.8);
  });

  it('bpwFor agrees with the table for every quantization', () => {
    for (const quant of QUANT_ORDER) {
      expect(bpwFor(quant)).toBe(QUANT_BPW[quant]);
    }
  });

  it('bpw is strictly decreasing from FP16 to Q4_K_M', () => {
    for (let i = 1; i < QUANT_ORDER.length; i++) {
      const previous = QUANT_ORDER[i - 1];
      const current = QUANT_ORDER[i];
      expect(previous).toBeDefined();
      expect(current).toBeDefined();
      if (previous !== undefined && current !== undefined) {
        expect(bpwFor(current)).toBeLessThan(bpwFor(previous));
      }
    }
  });
});
