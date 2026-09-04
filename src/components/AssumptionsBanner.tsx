import { IconInfo } from './Icon';

interface AssumptionsBannerProps {
  readonly version: string;
}

export function AssumptionsBanner({ version }: AssumptionsBannerProps) {
  return (
    <aside className="assumptions-banner" role="note">
      <IconInfo width={18} height={18} />
      <div>
        <p className="assumptions-title">Hypothèses et limites</p>
        <ul className="assumptions-list">
          <li>Ordre de grandeur calibré sur llama.cpp : ±25-30 % en débit pour les modèles denses sur GPU grand public.</li>
          <li>Exceptions connues (prédictions optimistes) : modèles MoE à contexte court, GPU datacenter sur petits modèles.</li>
          <li>La VRAM requise n’est jamais sous-estimée : elle est sur-estimée de 25-50 % (choix conservateur).</li>
          <li>Le modèle de parallélisme tensoriel est idéalisé ; outil d’aide à la décision d’achat, pas un benchmark.</li>
          <li>Prix relevés le {version} (EUR TTC).</li>
        </ul>
      </div>
    </aside>
  );
}
