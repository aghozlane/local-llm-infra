/**
 * Pure calculation engine for the LLM hardware sizing predictor.
 *
 * Zero dependency on React, the DOM or the network: everything exported here
 * is either a type, a documented constant or a pure function.
 * See docs/superpowers/specs/2026-09-04-llm-hardware-predictor-design.md (§4)
 * for the formula reference.
 */
export * from './types';
export * from './quantization';
export * from './kv';
export * from './formulas';
export * from './recommendations';
