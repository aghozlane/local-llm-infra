import type { Recommendation } from '../engine/recommendations';
import { formatCurrencyEur, formatGib, formatSeconds, formatTps } from '../lib/format';
import { BomTable } from './BomTable';
import { Tooltip } from './Tooltip';

interface RecommendationCardProps {
  readonly recommendation: Recommendation;
}

const ROLE_LABELS = {
  cheapest: 'La moins chère',
  balanced: 'L’équilibrée',
  comfortable: 'La confortable',
} as const;

export function RecommendationCard({ recommendation }: RecommendationCardProps) {
  const { role, pick } = recommendation;
  const sizing = pick.sizing;
  const stationUnitCount = pick.kind === 'station' ? pick.unitCount : 1;

  const configLine =
    pick.kind === 'gpu-platform'
      ? `${pick.gpuCount} × ${pick.gpu.name} — ${pick.platform.name}`
      : stationUnitCount > 1
        ? `${stationUnitCount} × ${pick.station.name}`
        : pick.station.name;

  const ramGib =
    pick.kind === 'gpu-platform' ? pick.ramGib : pick.station.unifiedMemoryGib * stationUnitCount;
  const gpuCount = pick.kind === 'gpu-platform' ? pick.gpuCount : stationUnitCount;

  return (
    <article className="card recommendation-card">
      <div className="recommendation-header">
        <span className="recommendation-role">{ROLE_LABELS[role]}</span>
        <h3 className="recommendation-config">{configLine}</h3>
        <span className="recommendation-price">{formatCurrencyEur(pick.totalEur)}</span>
      </div>

      <div className="recommendation-metrics">
        <div className="metric">
          <span className="metric-label">GPU</span>
          <span className="metric-value">{gpuCount}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Mémoire</span>
          <span className="metric-value">{formatGib(ramGib)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">Débit / utilisateur</span>
          <span className="metric-value">{formatTps(sizing.tpsPerUser)}</span>
        </div>
        <div className="metric">
          <span className="metric-label">
            <Tooltip label="Time To First Token : temps entre l'envoi de la requête et le premier token généré.">
              TTFT
            </Tooltip>
          </span>
          <span className="metric-value">{formatSeconds(sizing.ttftSeconds)}</span>
        </div>
      </div>

      <details className="bom-details">
        <summary className="bom-summary">Détail du devis</summary>
        <BomTable bom={pick.bom} />
      </details>
    </article>
  );
}
