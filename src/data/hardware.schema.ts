/**
 * Types et validation du catalogue matériel (étape 4 du plan).
 *
 * Le moteur (src/engine) ne définit pas encore `HardwareCatalog` (étape 2) :
 * ces types sont donc LOCAUX à src/data et seront fusionnés avec le moteur
 * lors de l'intégration (étape 8). Ils ne dépendent ni de React, ni du DOM,
 * ni du réseau.
 *
 * Conventions d'unités (alignées sur src/engine/types.ts — IMPORTANT) :
 * - Les CAPACITÉS mémoire (VRAM, RAM, mémoire unifiée) sont en Gio (GiB,
 *   base 2). Une carte vendue « 24 Go » contient physiquement 24 Gio.
 * - Les BANDES PASSANTE mémoire sont décimales (Go/s = GB/s = 10^9 octets/s),
 *   comme annoncées dans les fiches constructeurs.
 * - Les PUISSANCES de calcul sont en FLOP/s FP16 « dense » (sans éparsité),
 *   dans la convention communément citée pour l'inférence LLM
 *   (ex. RTX 4090 ≈ 165 TFLOPS, A100 ≈ 312 TFLOPS, H100 ≈ 989 TFLOPS SXM).
 * - Les PRIX sont en EUR TTC (marché européen), relevés le 2026-09-04.
 *
 * Le champ `category` a des valeurs en anglais (code), l'interface utilisateur
 * les traduit en français. Le segment « GP » du design = `consumer`.
 */

export type HardwareCategory = 'gpu' | 'station' | 'platform' | 'ram';

/** Segment GPU du design §5.1 : GP = consumer, Pro = pro, Datacenter = datacenter. */
export type GpuSegment = 'consumer' | 'pro' | 'datacenter';

/** 'listed' = prix relevé sur la source indiquée ; 'estimate' = valeur dérivée (voir notes). */
export type PriceQuality = 'listed' | 'estimate';

/** Champs communs à toutes les entrées du catalogue. */
export interface HardwareItemBase {
  /** Identifiant stable, kebab-case (ex. 'rtx-4090'). */
  readonly id: string;
  /** Nom affichable (français, format marketing). */
  readonly name: string;
  /** Marque / fabricant (NVIDIA, AMD, Apple, Lenovo, G.Skill...). */
  readonly brand: string;
  /** Famille d'équipement. */
  readonly category: HardwareCategory;
  /** Prix relevé en EUR TTC. */
  readonly priceEur: number;
  /** Date du relevé de prix, au format ISO 'YYYY-MM-DD'. */
  readonly priceDate: string;
  /** Qualité du prix : relevé ou estimation dérivée. */
  readonly priceQuality: PriceQuality;
  /** URL du relevé de prix (source obligatoire, même pour une estimation). */
  readonly source: string;
  /** Notes : contexte du relevé, dérivations, avertissements, concordances. */
  readonly notes?: string;
}

/** Carte graphique (design §5.1) : VRAM ~93 % utilisable, BW Go/s, FLOPS FP16, TDP, prix. */
export interface CatalogGpu extends HardwareItemBase {
  readonly category: 'gpu';
  readonly segment: GpuSegment;
  /** VRAM commercialisée, en Gio (base 2). */
  readonly vramGib: number;
  /** Bande passante mémoire, en Go/s (décimal). */
  readonly bwGbps: number;
  /** Puissance de calcul FP16 dense, en FLOP/s. */
  readonly flopsFp16: number;
  /** TDP (enveloppe thermique de la carte), en watts. */
  readonly tdpWatts: number;
  /** Part de VRAM utilisable (constante du design ≈ 0.93). */
  readonly usableVramRatio: number;
}

/** Station IA compacte à mémoire unifiée (design §5.2) : compute intégré, pas de VRAM dédiée. */
export interface CatalogStation extends HardwareItemBase {
  readonly category: 'station';
  /** Mémoire unifiée totale (partagée CPU+GPU), en Gio. */
  readonly unifiedMemoryGib: number;
  /** Bande passante mémoire, en Go/s (décimal). */
  readonly bwGbps: number;
  /** Puissance de calcul FP16 dense, en FLOP/s. */
  readonly flopsFp16: number;
}

/** Plateforme GPU (design §5.3) : châssis complet HORS GPU et HORS RAM (la BOM les ajoute). */
export interface CatalogPlatform extends HardwareItemBase {
  readonly category: 'platform';
  /** Nombre maximum de GPU installables. */
  readonly maxGpuSlots: number;
  /** RAM système maximum supportée, en Gio. */
  readonly maxRamGib: number;
}

/** Kit de RAM DDR5 (design §5.4). Prix relevé au kit + prix au Go dérivé. */
export interface RamKit extends HardwareItemBase {
  readonly category: 'ram';
  /** Capacité du kit, en Gio. */
  readonly capacityGib: number;
  /** Prix au Gio dérivé (priceEur / capacityGib, arrondi à 2 décimales). */
  readonly pricePerGibEur: number;
}

/** Une entrée quelconque du catalogue. */
export type HardwareItem = CatalogGpu | CatalogStation | CatalogPlatform | RamKit;

/** Catalogue matériel versionné (JSON : version = date du relevé). */
export interface HardwareCatalog {
  /** Version du relevé = date ISO 'YYYY-MM-DD' (ex. '2026-09-04'). */
  readonly version: string;
  /** Entrées du catalogue. */
  readonly items: readonly HardwareItem[];
}

const ISO_DATE = /^\d{4}-\d{2}-\d{2}$/;
const ID_PATTERN = /^[a-z0-9-]+$/;

function isPositiveNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0;
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.length > 0;
}

/**
 * Valide l'intégralité du catalogue (types ET invariants métier).
 * Retourne la liste des erreurs trouvées ; vide = catalogue valide.
 * Conçu pour la donnée désérialisée du JSON (les champs peuvent être
 * absents/mal typés à l'exécution malgré les types statiques).
 */
export function validateHardwareCatalog(catalog: HardwareCatalog): string[] {
  const errors: string[] = [];

  if (!isNonEmptyString(catalog.version) || !ISO_DATE.test(catalog.version)) {
    return ['version: date ISO YYYY-MM-DD attendue'];
  }
  if (!Array.isArray(catalog.items)) {
    return ['items: tableau attendu'];
  }
  if (catalog.items.length === 0) {
    errors.push('items: catalogue vide');
  }

  const seenIds = new Set<string>();
  for (const item of catalog.items) {
    const label = `item ${isNonEmptyString(item.id) ? item.id : '(id manquant)'}`;

    if (!isNonEmptyString(item.id) || !ID_PATTERN.test(item.id)) {
      errors.push(`${label}: id kebab-case requis`);
    } else if (seenIds.has(item.id)) {
      errors.push(`${label}: id en doublon`);
    } else {
      seenIds.add(item.id);
    }

    if (!isNonEmptyString(item.name)) errors.push(`${label}: name requis`);
    if (!isNonEmptyString(item.brand)) errors.push(`${label}: brand requis`);
    if (!isPositiveNumber(item.priceEur)) errors.push(`${label}: priceEur > 0 requis`);
    if (!isNonEmptyString(item.priceDate) || !ISO_DATE.test(item.priceDate)) {
      errors.push(`${label}: priceDate ISO requise`);
    }
    if (item.priceQuality !== 'listed' && item.priceQuality !== 'estimate') {
      errors.push(`${label}: priceQuality listed|estimate requis`);
    }
    if (!isNonEmptyString(item.source) || !/^https:\/\//.test(item.source)) {
      errors.push(`${label}: source https requise`);
    }
    if (item.notes !== undefined && typeof item.notes !== 'string') {
      errors.push(`${label}: notes doit être une chaîne`);
    }

    if (item.category === 'gpu') {
      if (item.segment !== 'consumer' && item.segment !== 'pro' && item.segment !== 'datacenter') {
        errors.push(`${label}: segment consumer|pro|datacenter requis`);
      }
      if (!isPositiveNumber(item.vramGib)) errors.push(`${label}: vramGib > 0 requis`);
      if (!isPositiveNumber(item.bwGbps)) errors.push(`${label}: bwGbps > 0 requis`);
      if (!isPositiveNumber(item.flopsFp16)) errors.push(`${label}: flopsFp16 > 0 requis`);
      if (!isPositiveNumber(item.tdpWatts)) errors.push(`${label}: tdpWatts > 0 requis`);
      if (
        typeof item.usableVramRatio !== 'number' ||
        !Number.isFinite(item.usableVramRatio) ||
        item.usableVramRatio <= 0 ||
        item.usableVramRatio > 1
      ) {
        errors.push(`${label}: usableVramRatio attendu dans (0, 1]`);
      }
    } else if (item.category === 'station') {
      if (!isPositiveNumber(item.unifiedMemoryGib)) {
        errors.push(`${label}: unifiedMemoryGib > 0 requis`);
      }
      if (!isPositiveNumber(item.bwGbps)) errors.push(`${label}: bwGbps > 0 requis`);
      if (!isPositiveNumber(item.flopsFp16)) errors.push(`${label}: flopsFp16 > 0 requis`);
    } else if (item.category === 'platform') {
      if (
        typeof item.maxGpuSlots !== 'number' ||
        !Number.isInteger(item.maxGpuSlots) ||
        item.maxGpuSlots < 1
      ) {
        errors.push(`${label}: maxGpuSlots entier >= 1 requis`);
      }
      if (!isPositiveNumber(item.maxRamGib)) errors.push(`${label}: maxRamGib > 0 requis`);
    } else if (item.category === 'ram') {
      if (!isPositiveNumber(item.capacityGib)) errors.push(`${label}: capacityGib > 0 requis`);
      if (
        isPositiveNumber(item.capacityGib) &&
        isPositiveNumber(item.priceEur) &&
        isPositiveNumber(item.pricePerGibEur) &&
        Math.abs(item.pricePerGibEur - item.priceEur / item.capacityGib) > 0.01
      ) {
        errors.push(`${label}: pricePerGibEur incohérent avec priceEur/capacityGib`);
      } else if (!isPositiveNumber(item.pricePerGibEur)) {
        errors.push(`${label}: pricePerGibEur > 0 requis`);
      }
    } else {
      errors.push(`${label}: category gpu|station|platform|ram requise`);
    }
  }

  return errors;
}
