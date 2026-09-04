/**
 * Helpers for formatting numbers, currency and memory figures in French.
 *
 * All formatting uses Intl.NumberFormat with locale 'fr-FR' so decimals use
 * commas, thousands use narrow non-breaking spaces, and currency is EUR.
 */

const numberFmt = new Intl.NumberFormat('fr-FR');
const currencyFmt = new Intl.NumberFormat('fr-FR', {
  style: 'currency',
  currency: 'EUR',
  maximumFractionDigits: 2,
});

const percentFmt = new Intl.NumberFormat('fr-FR', {
  style: 'percent',
  maximumFractionDigits: 0,
});

function safeNumber(value: number): number {
  return Number.isFinite(value) ? value : 0;
}

/** Format a generic integer or decimal number for French readers. */
export function formatNumber(value: number): string {
  return numberFmt.format(safeNumber(value));
}

/** Format a number with up to 2 decimal places. */
export function formatDecimal(value: number, maximumFractionDigits = 2): string {
  const fmt = new Intl.NumberFormat('fr-FR', {
    maximumFractionDigits,
    minimumFractionDigits: 0,
  });
  return fmt.format(safeNumber(value));
}

/** Format an amount in EUR. */
export function formatCurrencyEur(value: number): string {
  return currencyFmt.format(safeNumber(value));
}

/** Format a whole number. */
export function formatInteger(value: number): string {
  return numberFmt.format(Math.round(safeNumber(value)));
}

/** Format a fraction as a percentage (0.2 -> 20 %). */
export function formatPercent(value: number): string {
  return percentFmt.format(safeNumber(value));
}

/** Format a memory figure in GiB with the French marketing label "Go". */
export function formatGib(value: number): string {
  return `${formatDecimal(value)} Go`;
}

/** Format a throughput figure in tokens per second. */
export function formatTps(value: number): string {
  return `${formatDecimal(value)} tok/s`;
}

/** Format a latency in seconds, keeping 2 decimals when small. */
export function formatSeconds(value: number): string {
  return `${formatDecimal(value)} s`;
}
