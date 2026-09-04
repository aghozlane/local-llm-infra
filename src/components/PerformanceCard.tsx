import type { SizingResult } from '../engine/types';
import { formatDecimal, formatTps } from '../lib/format';
import { IconCheck, IconCross, IconWarning } from './Icon';

interface PerformanceCardProps {
  readonly sizing: SizingResult;
}

function TargetBadge({ met, label }: { readonly met: boolean | null; readonly label: string }) {
  if (met === null) {
    return (
      <span className="badge badge-neutral" title={`Aucune cible ${label}`}>
        {label} —
      </span>
    );
  }
  if (met) {
    return (
      <span className="badge badge-success">
        <IconCheck width={12} height={12} />
        {label}
      </span>
    );
  }
  return (
    <span className="badge badge-error">
      <IconCross width={12} height={12} />
      {label}
    </span>
  );
}

export function PerformanceCard({ sizing }: PerformanceCardProps) {
  return (
    <article className="card">
      <h2 className="card-title">Performances</h2>

      <div className="metrics-grid">
        <div className="metric">
          <span className="metric-label">TTFT</span>
          <span className="metric-value">{formatDecimal(sizing.ttftSeconds)} s</span>
        </div>
        <div className="metric">
          <span className="metric-label">Débit / utilisateur</span>
          <span className="metric-value">{formatTps(sizing.tpsPerUser)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Débit agrégé</span>
          <span className="metric-value">{formatTps(sizing.aggregateToksPerSec)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Charge offerte</span>
          <span className="metric-value">{formatTps(sizing.offeredLoadToksPerSec)}</span>
          {sizing.loadSupported ? (
            <span className="badge badge-success">
              <IconCheck width={12} height={12} />
              absorbée
            </span>
          ) : (
            <span className="badge badge-error">
              <IconCross width={12} height={12} />
              non absorbée
            </span>
          )}
        </div>
      </div>

      <div className="target-badges cluster">
        <TargetBadge met={sizing.ttftTargetMet} label="TTFT" />
        <TargetBadge met={sizing.tpsTargetMet} label="TPS" />
      </div>

      {sizing.contextTooSmall && (
        <div className="inline-warning" role="alert">
          <IconWarning width={16} height={16} />
          <p>
            Contexte max inférieur à la somme in + out. Le moteur suppose malgré tout la
            longueur demandée pour le calcul.
          </p>
        </div>
      )}
    </article>
  );
}
