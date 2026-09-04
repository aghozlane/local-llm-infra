/**
 * KV cache calculations (plan formula, verbatim):
 *
 *   KV cache (bytes) = 2 (K+V) x L x kv_h x d_h x 2 (FP16)
 *                      x max_context x simultaneous_sequences
 *
 * The per-token factor (everything before the context and sequence factors)
 * is exposed separately because the decode-throughput roofline reuses it as
 * "kv_bytes_per_token".
 */

/** Number of projection matrices stored per layer: K and V. */
export const KV_PROJECTIONS = 2;

/** KV cache elements are stored in FP16: 2 bytes each. */
export const KV_BYTES_PER_ELEMENT = 2;

/**
 * KV cache bytes per token, for ONE sequence:
 * 2 (K+V) x L x kv_h x d_h x 2 (FP16).
 */
export function kvBytesPerToken(
  numLayers: number,
  numKvHeads: number,
  headDim: number,
): number {
  return KV_PROJECTIONS * numLayers * numKvHeads * headDim * KV_BYTES_PER_ELEMENT;
}

/** Total KV cache bytes: per-token footprint x max_context x simultaneous. */
export function kvCacheBytes(
  bytesPerToken: number,
  contextMax: number,
  simultaneous: number,
): number {
  return bytesPerToken * contextMax * simultaneous;
}
