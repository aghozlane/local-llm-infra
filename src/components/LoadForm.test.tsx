import { describe, it, expect } from 'vitest';
import { CONTEXT_OPTIONS } from './LoadForm';

describe('CONTEXT_OPTIONS', () => {
  it('includes the 1M-token option', () => {
    expect(CONTEXT_OPTIONS).toContain(1048576);
  });

  it('follows the power-of-two progression', () => {
    for (let i = 1; i < CONTEXT_OPTIONS.length; i++) {
      expect(CONTEXT_OPTIONS[i]).toBe(CONTEXT_OPTIONS[i - 1]! * 2);
    }
  });

  it('has exactly nine options', () => {
    expect(CONTEXT_OPTIONS).toHaveLength(9);
  });
});
