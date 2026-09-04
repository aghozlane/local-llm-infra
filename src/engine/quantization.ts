/**
 * Quantization table: effective bits per weight (bpw) per format.
 *
 * The values are VERBATIM from the validated plan (section "Formules &
 * constantes"). They approximate llama.cpp GGUF effective footprints
 * (e.g. Q8_0 = 8.5 because of its per-block scale fields).
 * DO NOT derive or change these values - calibration is a separate plan step.
 */
import type { QuantName } from './types';

/** Effective bits-per-weight for each supported quantization format. */
export const QUANT_BPW: Readonly<Record<QuantName, number>> = {
  FP16: 16,
  Q8_0: 8.5,
  Q6_K: 6.6,
  Q5_K_M: 5.7,
  Q4_K_M: 4.8,
};

/** Canonical quantization list, from most to least precise. */
export const QUANT_ORDER: readonly QuantName[] = [
  'FP16',
  'Q8_0',
  'Q6_K',
  'Q5_K_M',
  'Q4_K_M',
];

/** Effective bits-per-weight for a quantization format. */
export function bpwFor(quant: QuantName): number {
  return QUANT_BPW[quant];
}
