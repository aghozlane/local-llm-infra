import { describe, expect, it } from 'vitest';
import {
  formatCurrencyEur,
  formatDecimal,
  formatGib,
  formatInteger,
  formatNumber,
  formatPercent,
  formatSeconds,
  formatTps,
} from './format';

describe('format helpers', () => {
  it('formats integers with fr-FR grouping', () => {
    expect(formatNumber(1234567)).toBe('1\u202f234\u202f567');
  });

  it('formats decimals with a comma', () => {
    expect(formatDecimal(1234.56)).toBe('1\u202f234,56');
  });

  it('formats currency in EUR', () => {
    expect(formatCurrencyEur(1234.5)).toBe('1\u202f234,50\u00a0€');
  });

  it('formats integers by rounding', () => {
    expect(formatInteger(1234.56)).toBe('1\u202f235');
  });

  it('formats fractions as percentages', () => {
    expect(formatPercent(0.2)).toBe('20\u00a0%');
  });

  it('formats memory in Go', () => {
    expect(formatGib(16.5)).toBe('16,5 Go');
  });

  it('formats throughput', () => {
    expect(formatTps(42.5)).toBe('42,5 tok/s');
  });

  it('formats seconds', () => {
    expect(formatSeconds(1.23)).toBe('1,23 s');
  });

  it('falls back to zero for non-finite values', () => {
    expect(formatNumber(NaN)).toBe('0');
    expect(formatNumber(Infinity)).toBe('0');
  });
});
