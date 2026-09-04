/**
 * Sélection matériel & BOM (étape 5 du plan).
 *
 * Module pur : (ModelSpec, Scenario, HardwareCatalog) → RecommendationResult.
 * Aucune dépendance React / DOM / réseau. Réutilise les formules de
 * formulas.ts (modèle de parallélisme tensoriel idéalisé : G GPU partagent
 * poids et KV et apportent G × bande passante / G × FLOPS).
 *
 * Logique de sélection (plan §5.5) :
 * 1. Énumérer : chaque GPU du catalogue × quantité (quantité = nb GPU imposé
 *    par la VRAM, ≤ slots de chaque plateforme) + les stations compactes
 *    (unité simple, pas de multi-unité en v1).
 * 2. Filtrer : VRAM suffisante **avec marge** (le gpuCount de SizingResult
 *    inclut déjà la marge de sécurité et le ratio ~93 % de VRAM utilisable),
 *    cibles TTFT et débit par utilisateur (dures, si fournies), RAM système
 *    ≤ maximum de la plateforme.
 * 3. Classer par prix total = n × GPU + plateforme + RAM.
 * 4. Renvoyer jusqu'à 3 recommandations : la **moins chère** (première),
 *    l'**équilibrée** (médiane du classement) et la **confortable** (dernière).
 * 5. Aucune combinaison viable → statut 'aucune' + raisons explicites
 *    (réduire contexte / simultanés, quantifier davantage, etc.).
 *
 * RAM : le design prévoit un prix au Go (~5 €/Go DDR5 ECC) : quantité
 * achetée = ceil(systemRamGib), tarif = prix au Gio du kit le moins cher du
 * catalogue. Un catalogue sans aucun kit RAM est un bug de donnée : la
 * fonction lève une erreur (pas de repli silencieux).
 */
import { MAX_GPUS_V1, sizeForHardware } from './formulas';
import type { GpuSpec, ModelSpec, Scenario, SizingResult } from './types';
import type {
  CatalogGpu,
  CatalogPlatform,
  CatalogStation,
  HardwareCatalog,
  RamKit,
} from '../data/hardware.schema';

/** Ligne de la BOM (style devis). */
export interface BomLine {
  /** Libellé français de la ligne. */
  readonly label: string;
  /** Quantité (nb GPU, nb plateforme, nb Go de RAM). */
  readonly quantity: number;
  /** Prix unitaire en EUR TTC. */
  readonly unitPriceEur: number;
  /** Ligne totale en EUR TTC (quantity × unitPriceEur, arrondie au centime). */
  readonly totalEur: number;
}

/** Configuration GPU + plateforme (châssis complet). */
export interface GpuPlatformPick {
  readonly kind: 'gpu-platform';
  readonly gpu: CatalogGpu;
  /** Nombre de GPU identiques à installer. */
  readonly gpuCount: number;
  readonly platform: CatalogPlatform;
  /** Quantité de RAM achetée en Gio (ceil de systemRamGib). */
  readonly ramGib: number;
  readonly ramCostEur: number;
  /** Prix total de la configuration en EUR TTC. */
  readonly totalEur: number;
  readonly bom: readonly BomLine[];
  /** Résultat de dimensionnement complet de la configuration. */
  readonly sizing: SizingResult;
}

/** Configuration station IA compacte (mémoire unifiée, unité simple). */
export interface StationPick {
  readonly kind: 'station';
  readonly station: CatalogStation;
  /** Prix total en EUR TTC (machine seule, mémoire intégrée). */
  readonly totalEur: number;
  readonly bom: readonly BomLine[];
  /** Résultat de dimensionnement complet de la configuration. */
  readonly sizing: SizingResult;
}

export type HardwarePick = GpuPlatformPick | StationPick;

/** Rôle de la recommandation dans le trio (l'UI le traduit en français). */
export type RecommendationRole = 'cheapest' | 'balanced' | 'comfortable';

export interface Recommendation {
  readonly role: RecommendationRole;
  readonly pick: HardwarePick;
}

export interface RecommendationResult {
  /** 'viable' : au moins une configuration ; 'aucune' : rien ne passe les filtres. */
  readonly status: 'viable' | 'aucune';
  /** 0 à 3 recommandations (uniquement quand status = 'viable'). */
  readonly recommendations: readonly Recommendation[];
  /** Raisons explicites en français quand status = 'aucune' (sinon vide). */
  readonly reasons: readonly string[];
}

/** Arrondi au centime (EUR TTC). */
function roundEur(value: number): number {
  return Math.round(value * 100) / 100;
}

/** Identifiant de déduplication d'une configuration. */
function pickId(pick: HardwarePick): string {
  return pick.kind === 'gpu-platform'
    ? `${pick.gpu.id}+${pick.platform.id}`
    : `station:${pick.station.id}`;
}

/** Nombre de GPU d'une configuration (station = unité simple). */
function pickGpuCount(pick: HardwarePick): number {
  return pick.kind === 'gpu-platform' ? pick.gpuCount : 1;
}

/** Une station est vue comme un GPU unique à mémoire unifiée. */
function stationAsGpuSpec(station: CatalogStation): GpuSpec {
  return {
    name: station.name,
    vramGib: station.unifiedMemoryGib,
    bwGbps: station.bwGbps,
    flopsFp16: station.flopsFp16,
  };
}

/** Portes dures de cibles : null = OK, sinon la cible non atteinte. */
function targetFailure(sizing: SizingResult): 'ttft' | 'tps' | null {
  if (sizing.ttftTargetMet === false) return 'ttft';
  if (sizing.tpsTargetMet === false) return 'tps';
  return null;
}

interface RejectionCounters {
  vram: number;
  ramCap: number;
  ttft: number;
  tps: number;
}

function buildReasons(rejected: RejectionCounters): string[] {
  const reasons: string[] = [];
  if (rejected.vram > 0) {
    reasons.push(
      'VRAM requise (marge comprise) supérieure au catalogue v1 (8 GPU maximum, slots plateforme limités) : réduisez le contexte max ou le nombre de séquences simultanées, ou passez à une quantification plus agressive (Q5_K_M, Q4_K_M).',
    );
  }
  if (rejected.ramCap > 0) {
    reasons.push(
      'RAM système requise au-delà du maximum supporté par les plateformes du catalogue : réduisez la configuration GPU.',
    );
  }
  if (rejected.ttft > 0) {
    reasons.push(
      'Cible TTFT non atteinte sur aucune configuration : réduisez la longueur d’entrée ou la charge simultanée, ou visez des GPU plus performants en FLOPS FP16.',
    );
  }
  if (rejected.tps > 0) {
    reasons.push(
      'Débit cible par utilisateur non atteint sur aucune configuration : réduisez le contexte moyen ou les séquences simultanées, ou visez des GPU à bande passante mémoire plus élevée.',
    );
  }
  if (reasons.length === 0) {
    reasons.push(
      'Aucune combinaison du catalogue ne satisfait le scénario : réduisez le contexte max, les séquences simultanées, ou quantifiez davantage.',
    );
  }
  return reasons;
}

/**
 * Sélectionne jusqu'à 3 configurations viables (moins chère / équilibrée /
 * confortable) et chiffre leur BOM, à partir du catalogue matériel.
 *
 * @throws Error si le catalogue ne contient aucun kit RAM (chiffrage BOM
 *   impossible) — bug de donnée, pas de repli.
 */
export function recommendHardware(
  spec: ModelSpec,
  scenario: Scenario,
  catalog: HardwareCatalog,
): RecommendationResult {
  const gpus = catalog.items.filter((item): item is CatalogGpu => item.category === 'gpu');
  const stations = catalog.items.filter((item): item is CatalogStation => item.category === 'station');
  const platforms = catalog.items.filter((item): item is CatalogPlatform => item.category === 'platform');
  const ramKits = catalog.items.filter((item): item is RamKit => item.category === 'ram');

  if (ramKits.length === 0) {
    throw new Error('Catalogue matériel : aucun kit RAM — BOM impossible à chiffrer');
  }
  const ramRateEurPerGib = roundEur(Math.min(...ramKits.map((kit) => kit.pricePerGibEur)));

  const rejected: RejectionCounters = { vram: 0, ramCap: 0, ttft: 0, tps: 0 };
  const candidates: HardwarePick[] = [];

  // --- Combinaisons GPU + plateforme ---
  for (const gpu of gpus) {
    // sizeForHardware est identique pour toutes les plateformes d'un même GPU :
    // la seule différence entre plateformes est la capacité (slots, RAM max).
    const sizing = sizeForHardware(spec, scenario, gpu);
    for (const platform of platforms) {
      if (sizing.gpuCount > platform.maxGpuSlots || sizing.gpuCount > MAX_GPUS_V1) {
        rejected.vram += 1;
        continue;
      }
      if (sizing.systemRamGib > platform.maxRamGib) {
        rejected.ramCap += 1;
        continue;
      }
      const failure = targetFailure(sizing);
      if (failure !== null) {
        rejected[failure] += 1;
        continue;
      }
      const ramGib = Math.ceil(sizing.systemRamGib);
      const ramCostEur = roundEur(ramGib * ramRateEurPerGib);
      const gpuCostEur = roundEur(sizing.gpuCount * gpu.priceEur);
      const totalEur = roundEur(gpuCostEur + platform.priceEur + ramCostEur);
      const bom: BomLine[] = [
        { label: gpu.name, quantity: sizing.gpuCount, unitPriceEur: gpu.priceEur, totalEur: gpuCostEur },
        { label: platform.name, quantity: 1, unitPriceEur: platform.priceEur, totalEur: platform.priceEur },
        { label: 'RAM DDR5 ECC', quantity: ramGib, unitPriceEur: ramRateEurPerGib, totalEur: ramCostEur },
      ];
      candidates.push({
        kind: 'gpu-platform',
        gpu,
        gpuCount: sizing.gpuCount,
        platform,
        ramGib,
        ramCostEur,
        totalEur,
        bom,
        sizing,
      });
    }
  }

  // --- Stations compactes (unité simple) ---
  for (const station of stations) {
    const sizing = sizeForHardware(spec, scenario, stationAsGpuSpec(station));
    if (sizing.gpuCount > 1) {
      rejected.vram += 1;
      continue;
    }
    const failure = targetFailure(sizing);
    if (failure !== null) {
      rejected[failure] += 1;
      continue;
    }
    const totalEur = roundEur(station.priceEur);
    const bom: BomLine[] = [
      { label: station.name, quantity: 1, unitPriceEur: station.priceEur, totalEur },
    ];
    candidates.push({ kind: 'station', station, totalEur, bom, sizing });
  }

  // --- Aucune configuration viable ---
  if (candidates.length === 0) {
    return { status: 'aucune', recommendations: [], reasons: buildReasons(rejected) };
  }

  // --- Classement par prix total (doublons : nb GPU puis libellé) ---
  const sorted = [...candidates].sort(
    (a, b) =>
      a.totalEur - b.totalEur ||
      pickGpuCount(a) - pickGpuCount(b) ||
      (a.bom[0]?.label ?? '').localeCompare(b.bom[0]?.label ?? ''),
  );

  const first = sorted[0];
  const middle = sorted[Math.floor((sorted.length - 1) / 2)];
  const last = sorted[sorted.length - 1];
  if (first === undefined || middle === undefined || last === undefined) {
    // Invariant : sorted.length === candidates.length > 0.
    throw new Error('Sélection matériel : liste des candidats interne vide');
  }

  const recommendations: Recommendation[] = [];
  const addRole = (role: RecommendationRole, pick: HardwarePick): void => {
    if (!recommendations.some((entry) => pickId(entry.pick) === pickId(pick))) {
      recommendations.push({ role, pick });
    }
  };
  addRole('cheapest', first);
  addRole('balanced', middle);
  addRole('comfortable', last);

  return { status: 'viable', recommendations, reasons: [] };
}
