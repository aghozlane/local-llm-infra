import type { SizingResult } from '../engine/types';
import { formatDecimal, formatGib } from '../lib/format';
import { IconWarning } from './Icon';

interface MemoryCardProps {
  readonly sizing: SizingResult;
}

const RADIUS = 50;
const CIRCUMFERENCE = 2 * Math.PI * RADIUS;

interface Segment {
  readonly label: string;
  readonly value: number;
  readonly color: string;
}

export function MemoryCard({ sizing }: MemoryCardProps) {
  const total = sizing.weightGib + sizing.kvGib + sizing.overheadGib;
  const segments: Segment[] = [
    { label: 'Poids', value: sizing.weightGib, color: 'var(--accent-primary)' },
    { label: 'KV cache', value: sizing.kvGib, color: 'var(--status-info)' },
    { label: 'Overhead', value: sizing.overheadGib, color: 'var(--text-tertiary)' },
  ];

  let offset = 0;
  const arcs = segments.map((segment) => {
    const fraction = total > 0 ? segment.value / total : 0;
    const length = fraction * CIRCUMFERENCE;
    const arc = {
      ...segment,
      length,
      offset: -offset,
    };
    offset += length;
    return arc;
  });

  return (
    <article className="card">
      <h2 className="card-title">Mémoire</h2>

      <div className="donut-layout">
        <div className="donut-wrapper">
          <svg className="donut" viewBox="0 0 120 120" role="img" aria-label="Répartition VRAM">
            <g transform="rotate(-90 60 60)">
              {arcs.map((arc) => (
                <circle
                  key={arc.label}
                  cx="60"
                  cy="60"
                  r={RADIUS}
                  fill="none"
                  stroke={arc.color}
                  strokeWidth="18"
                  strokeLinecap="butt"
                  strokeDasharray={`${arc.length} ${CIRCUMFERENCE - arc.length}`}
                  strokeDashoffset={arc.offset}
                />
              ))}
            </g>
          </svg>
          <div className="donut-center">
            <span className="donut-value">{formatDecimal(sizing.requiredVramGib)}</span>
            <span className="donut-unit">Go requis</span>
            <span className="donut-note">(marge incluse)</span>
          </div>
        </div>

        <ul className="donut-legend">
          {segments.map((segment) => (
            <li key={segment.label} className="donut-legend-item">
              <span className="donut-swatch" style={{ background: segment.color }} />
              <span className="donut-label">{segment.label}</span>
              <span className="donut-amount">{formatGib(segment.value)}</span>
            </li>
          ))}
        </ul>
      </div>

      <div className="memory-meta cluster">
        <span className="badge badge-neutral">{sizing.gpuCount} GPU</span>
        {sizing.exceedsV1GpuCap && (
          <span className="badge badge-warning">
            <IconWarning width={12} height={12} />
            Hors catalogue v1 (&gt; 8 GPU)
          </span>
        )}
      </div>
    </article>
  );
}
