import { describe, expect, it } from 'vitest';
import { recommendHardware } from './index';
import type { HardwarePick, ModelSpec, Recommendation, Scenario } from './index';
import rawHardwareJson from '../data/hardware.json?raw';
import type {
  CatalogGpu,
  CatalogPlatform,
  CatalogStation,
  HardwareCatalog,
  RamKit,
} from '../data/hardware.schema';

// ---------------------------------------------------------------------------
// Fabrique de catalogues factices (aucune donnée du catalogue réel)
// ---------------------------------------------------------------------------

function gpu(
  id: string,
  name: string,
  vramGib: number,
  bwGbps: number,
  flopsFp16: number,
  priceEur: number,
): CatalogGpu {
  return {
    id,
    name,
    brand: 'Test',
    category: 'gpu',
    segment: 'consumer',
    priceEur,
    priceDate: '2026-09-04',
    priceQuality: 'listed',
    source: 'https://example.com/gpu',
    vramGib,
    bwGbps,
    flopsFp16,
    tdpWatts: 300,
    usableVramRatio: 0.93,
  };
}

function platform(id: string, name: string, maxGpuSlots: number, maxRamGib: number, priceEur: number): CatalogPlatform {
  return {
    id,
    name,
    brand: 'Test',
    category: 'platform',
    priceEur,
    priceDate: '2026-09-04',
    priceQuality: 'listed',
    source: 'https://example.com/platform',
    maxGpuSlots,
    maxRamGib,
  };
}

function station(id: string, name: string, unifiedMemoryGib: number, bwGbps: number, flopsFp16: number, priceEur: number): CatalogStation {
  return {
    id,
    name,
    brand: 'Test',
    category: 'station',
    priceEur,
    priceDate: '2026-09-04',
    priceQuality: 'listed',
    source: 'https://example.com/station',
    unifiedMemoryGib,
    bwGbps,
    flopsFp16,
  };
}

function ramKit(id: string, name: string, capacityGib: number, priceEur: number): RamKit {
  return {
    id,
    name,
    brand: 'Test',
    category: 'ram',
    priceEur,
    priceDate: '2026-09-04',
    priceQuality: 'listed',
    source: 'https://example.com/ram',
    capacityGib,
    pricePerGibEur: Math.round((priceEur / capacityGib) * 100) / 100,
  };
}

function catalog(items: HardwareCatalog['items']): HardwareCatalog {
  return { version: '2026-09-04', items };
}

// Catalogue « complet » : 3 GPU, 3 plateformes, 1 station, 2 kits RAM à 5 €/Go.
const CATALOGUE: HardwareCatalog = catalog([
  gpu('gpu-small', 'GPU Petit 16 Go', 16, 500, 1e14, 1_000),
  gpu('gpu-mid', 'GPU Moyen 32 Go', 32, 1_000, 2e14, 5_000),
  gpu('gpu-big', 'GPU Grand 80 Go', 80, 2_000, 7.5e14, 20_000),
  platform('plat-1', 'Plateforme 1 GPU', 1, 256, 500),
  platform('plat-2', 'Plateforme 2 GPU', 2, 512, 2_000),
  platform('plat-4', 'Plateforme 4 GPU', 4, 1024, 5_000),
  station('station-128', 'Station 128 Go', 128, 800, 3e14, 7_000),
  ramKit('ram-64', 'RAM 64 Go', 64, 320),
  ramKit('ram-128', 'RAM 128 Go', 128, 640),
]);

/** Catalogue réduit à une seule station (isole la logique multi-unités). */
const STATION_ONLY: HardwareCatalog = catalog([
  station('station-128', 'Station 128 Go', 128, 800, 3e14, 7_000),
  ramKit('ram-64', 'RAM 64 Go', 64, 320),
]);

/** Catalogue réel (src/data/hardware.json) pour les tests DGX B200/B300. */
const REAL_CATALOG: HardwareCatalog = JSON.parse(rawHardwareJson);

/** Sous-ensemble réel : DGX Spark + Mac Studio M5 Ultra + kits RAM réels. */
const REAL_STATIONS_SPARK_MAC: HardwareCatalog = {
  version: REAL_CATALOG.version,
  items: REAL_CATALOG.items.filter(
    (item) =>
      item.category === 'ram' ||
      item.id === 'nvidia-dgx-spark' ||
      item.id === 'mac-studio-m5-ultra-96',
  ),
};

/** MoE 30B (3,3 Md actifs/token) : même profil que le golden scenario. */
const SPEC_MOE_30B: ModelSpec = {
  totalParams: 30_000_000_000,
  numLayers: 48,
  numKvHeads: 4,
  headDim: 128,
  isMoe: true,
  numExpertsPerTok: 8,
  expertSize: 412_500_000,
};

const SCENARIO_GOLDEN: Scenario = {
  quant: 'Q4_K_M',
  simultaneous: 10,
  totalUsers: 30,
  reqPerUserPerMin: 3,
  inputTokens: 2000,
  outputTokens: 500,
  contextMax: 16_384,
  ttftTargetSec: 2,
  minTpsPerUser: 20,
};

// ---------------------------------------------------------------------------
// Modèle et scénario de référence (valeurs calculées à la main dans les asserts)
// ---------------------------------------------------------------------------

/** Dense 10B : poids Q4_K_M = 10e9 × 4.8 / 8 = 6e9 octets ≈ 5.588 Gio. */
const SPEC_10B: ModelSpec = {
  totalParams: 10_000_000_000,
  numLayers: 32,
  numKvHeads: 8,
  headDim: 128,
  isMoe: false,
};

/**
 * KV/token = 2 × 32 × 8 × 128 × 2 = 131 072 octets.
 * KV total = 131 072 × 4096 × 4 = 2 Gio.
 * VRAM requise = (5.588 + 2 + 2) × 1.2 ≈ 11.506 Gio → 1 GPU partout, RAM 64 Go.
 */
const SCENARIO: Scenario = {
  quant: 'Q4_K_M',
  simultaneous: 4,
  totalUsers: 10,
  reqPerUserPerMin: 1,
  inputTokens: 512,
  outputTokens: 128,
  contextMax: 4_096,
};

/** Dense 70B : poids Q4_K_M ≈ 39.08 Gio → VRAM ≈ 51.7 Gio. */
const SPEC_70B: ModelSpec = { ...SPEC_10B, totalParams: 70_000_000_000 };

/** Dense 500B FP16 : poids = 1 To ≈ 931 Gio → au-delà de tout le catalogue. */
const SPEC_500B_FP16: ModelSpec = { ...SPEC_10B, totalParams: 500_000_000_000 };

function scenarioOf(overrides: Partial<Scenario>): Scenario {
  return { ...SCENARIO, ...overrides };
}

// ---------------------------------------------------------------------------
// Aides d'assertion (indexation stricte : noUncheckedIndexedAccess)
// ---------------------------------------------------------------------------

function expectDefined<T>(value: T | undefined, label: string): T {
  if (value === undefined) throw new Error(`élément attendu absent : ${label}`);
  return value;
}

function roleOf(result: ReturnType<typeof recommendHardware>, role: string): HardwarePick {
  if (result.status !== 'viable') throw new Error(`statut attendu 'viable', obtenu 'aucune' (${result.reasons})`);
  const rec: Recommendation | undefined = result.recommendations.find((r) => r.role === role);
  if (rec === undefined) throw new Error(`recommandation '${role}' absente`);
  return rec.pick;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('recommendHardware — sélection moins chère / équilibrée / confortable', () => {
  it('produit 3 recommandations distinctes triées par prix sur le catalogue complet', () => {
    const result = recommendHardware(SPEC_10B, SCENARIO, CATALOGUE);
    expect(result.status).toBe('viable');
    expect(result.reasons).toEqual([]);
    expect(result.recommendations).toHaveLength(3);

    // Classement attendu (prix totaux = n × GPU + plateforme + RAM à 5 €/Go, RAM 64 Go = 320 €) :
    //  1820 small+plat-1, 3320 small+plat-2, 5820 mid+plat-1, 6320 small+plat-4,
    //  7000 station, 7320 mid+plat-2, 10320 mid+plat-4, 20820 big+plat-1,
    //  22320 big+plat-2, 25320 big+plat-4.
    const cheapest = roleOf(result, 'cheapest');
    const balanced = roleOf(result, 'balanced');
    const comfortable = roleOf(result, 'comfortable');

    if (cheapest.kind !== 'gpu-platform') throw new Error('cheapest doit être gpu-platform');
    expect(cheapest.gpu.id).toBe('gpu-small');
    expect(cheapest.platform.id).toBe('plat-1');
    expect(cheapest.gpuCount).toBe(1);
    expect(cheapest.totalEur).toBe(1_820);

    if (balanced.kind !== 'station') throw new Error('balanced doit être la station (médiane du classement)');
    expect(balanced.station.id).toBe('station-128');
    expect(balanced.unitCount).toBe(1);
    expect(balanced.totalEur).toBe(7_000);

    if (comfortable.kind !== 'gpu-platform') throw new Error('comfortable doit être gpu-platform');
    expect(comfortable.gpu.id).toBe('gpu-big');
    expect(comfortable.platform.id).toBe('plat-4');
    expect(comfortable.totalEur).toBe(25_320);

    // Le sizing embarqué est cohérent avec les formules (VRAM ≈ 11.506 Gio, RAM 64 Go).
    expect(cheapest.sizing.requiredVramGib).toBeCloseTo(11.505522536865234, 6);
    expect(cheapest.sizing.gpuCount).toBe(1);
    expect(cheapest.sizing.systemRamGib).toBe(64);
  });

  it('chiffre la BOM au Go de RAM (kit le moins cher) et arrondit au centime', () => {
    const result = recommendHardware(SPEC_10B, SCENARIO, CATALOGUE);
    const pick = roleOf(result, 'cheapest');
    if (pick.kind !== 'gpu-platform') throw new Error('attendu gpu-platform');

    expect(pick.bom).toHaveLength(3);
    const ramLine = expectDefined(pick.bom[2], 'ligne RAM');
    expect(ramLine.label).toBe('RAM DDR5 ECC');
    expect(ramLine.quantity).toBe(64);
    expect(ramLine.unitPriceEur).toBe(5);
    expect(ramLine.totalEur).toBe(320);

    // Invariant : la somme des lignes = prix total de la configuration.
    const bomTotal = Math.round(pick.bom.reduce((sum, line) => sum + line.totalEur, 0) * 100) / 100;
    expect(bomTotal).toBe(pick.totalEur);

    expect(pick.ramGib).toBe(64);
    expect(pick.ramCostEur).toBe(320);
  });
});

describe('recommendHardware — quantité de GPU et slots de plateforme', () => {
  it('impose le nb GPU imposé par la VRAM et respecte les slots des plateformes', () => {
    // SPEC_70B → VRAM ≈ 51.7 Gio : small = 4 GPU, mid = 2, big = 1, station = 1.
    // RAM = max(64, 1.5 × 51.7) = 77.54 → 78 Go achetés = 390 €.
    // Charge offerte 106,7 tok/s > agrégat d'une unité de station
    // (tps ≈ 16,1 × 4 séquences ≈ 64,3 tok/s) → la station passe à 2 unités
    // (2 × 7000 = 14 000 €) et n'est plus la moins chère.
    const result = recommendHardware(SPEC_70B, SCENARIO, CATALOGUE);
    expect(result.status).toBe('viable');

    // small+plat-1 (4 > 1) et small+plat-2 (4 > 2) et mid+plat-1 (2 > 1) sont exclus.
    // Classement : 9390 small+plat-4, 12390 mid+plat-2, 14000 station ×2,
    // 15390 mid+plat-4, 20890 big+plat-1, 22390 big+plat-2, 25390 big+plat-4 (7 candidats).
    const cheapest = roleOf(result, 'cheapest');
    if (cheapest.kind !== 'gpu-platform') throw new Error('cheapest doit être gpu-platform');
    expect(cheapest.gpu.id).toBe('gpu-small');
    expect(cheapest.platform.id).toBe('plat-4');
    expect(cheapest.gpuCount).toBe(4);
    expect(cheapest.totalEur).toBe(9_390);

    const balanced = roleOf(result, 'balanced');
    if (balanced.kind !== 'gpu-platform') throw new Error('balanced doit être gpu-platform');
    expect(balanced.gpu.id).toBe('gpu-mid');
    expect(balanced.platform.id).toBe('plat-4');
    expect(balanced.gpuCount).toBe(2);
    expect(balanced.ramGib).toBe(78);
    expect(balanced.ramCostEur).toBe(390);
    expect(balanced.totalEur).toBe(15_390);

    const comfortable = roleOf(result, 'comfortable');
    if (comfortable.kind !== 'gpu-platform') throw new Error('comfortable doit être gpu-platform');
    expect(comfortable.gpu.id).toBe('gpu-big');
    expect(comfortable.gpuCount).toBe(1);
    expect(comfortable.totalEur).toBe(25_390);

    // Invariant général : aucune recommandation ne dépasse les slots de sa plateforme.
    for (const rec of result.recommendations) {
      if (rec.pick.kind === 'gpu-platform') {
        expect(rec.pick.gpuCount).toBeLessThanOrEqual(rec.pick.platform.maxGpuSlots);
        expect(rec.pick.ramGib).toBeLessThanOrEqual(rec.pick.platform.maxRamGib);
      }
    }
  });
});

describe('recommendHardware — cibles dures (TTFT / débit)', () => {
  it('rejette tout quand la cible TTFT est inatteignable et explique pourquoi', () => {
    // TTFT big 1 GPU = 2 × 1e10 × 512 / (7.5e14 × 0.4) ≈ 0.0341 s > 0.01 s.
    const result = recommendHardware(SPEC_10B, scenarioOf({ ttftTargetSec: 0.01 }), CATALOGUE);
    expect(result.status).toBe('aucune');
    expect(result.recommendations).toHaveLength(0);
    expect(result.reasons.some((r) => r.includes('TTFT'))).toBe(true);
  });

  it('rejette tout quand la cible débit par utilisateur est inatteignable', () => {
    // Meilleur tps/user (big) ≈ 1.7e12 / (6e9 + 4 × 131072 × 576) ≈ 270 tok/s < 1e6.
    const result = recommendHardware(SPEC_10B, scenarioOf({ minTpsPerUser: 1_000_000 }), CATALOGUE);
    expect(result.status).toBe('aucune');
    expect(result.recommendations).toHaveLength(0);
    expect(result.reasons.some((r) => r.includes('Débit'))).toBe(true);
  });

  it('laisse passer les cibles atteintes (ttft 2 s, débit 10 tok/s)', () => {
    const result = recommendHardware(SPEC_10B, scenarioOf({ ttftTargetSec: 2, minTpsPerUser: 10 }), CATALOGUE);
    expect(result.status).toBe('viable');
    expect(result.recommendations).toHaveLength(3);
  });
});

describe('recommendHardware — aucune solution viable', () => {
  it('signale la VRAM hors catalogue v1 pour un modèle trop grand', () => {
    const result = recommendHardware(SPEC_500B_FP16, scenarioOf({ quant: 'FP16' }), CATALOGUE);
    expect(result.status).toBe('aucune');
    expect(result.recommendations).toHaveLength(0);
    expect(result.reasons.some((r) => r.includes('VRAM'))).toBe(true);
  });

  it('lève une erreur explicite si le catalogue n’a aucun kit RAM', () => {
    const sansRam = catalog([
      gpu('gpu-small', 'GPU Petit 16 Go', 16, 500, 1e14, 1_000),
      platform('plat-1', 'Plateforme 1 GPU', 1, 256, 500),
    ]);
    expect(() => recommendHardware(SPEC_10B, SCENARIO, sansRam)).toThrowError(/kit RAM/);
  });
});

describe('recommendHardware — déduplication des rôles', () => {
  it('retourne une seule recommandation quand une seule configuration est viable', () => {
    const mini = catalog([
      gpu('gpu-small', 'GPU Petit 16 Go', 16, 500, 1e14, 1_000),
      platform('plat-1', 'Plateforme 1 GPU', 1, 256, 500),
      ramKit('ram-64', 'RAM 64 Go', 64, 320),
    ]);
    const result = recommendHardware(SPEC_10B, SCENARIO, mini);
    expect(result.status).toBe('viable');
    expect(result.recommendations).toHaveLength(1);
    expect(result.recommendations[0]?.role).toBe('cheapest');
  });

  it('fusionne « équilibrée » sur « moins chère » quand seulement deux configurations sont viables', () => {
    const duo = catalog([
      gpu('gpu-small', 'GPU Petit 16 Go', 16, 500, 1e14, 1_000),
      platform('plat-1', 'Plateforme 1 GPU', 1, 256, 500),
      platform('plat-2', 'Plateforme 2 GPU', 2, 512, 2_000),
      ramKit('ram-64', 'RAM 64 Go', 64, 320),
    ]);
    const result = recommendHardware(SPEC_10B, SCENARIO, duo);
    expect(result.status).toBe('viable');
    // 2 candidats : cheapest = idx 0, balanced = idx floor(1/2) = 0 (dédup),
    // comfortable = idx 1.
    expect(result.recommendations).toHaveLength(2);
    const roles = result.recommendations.map((r) => r.role);
    expect(roles).toEqual(['cheapest', 'comfortable']);
    const cheapest = expectDefined(result.recommendations[0], 'cheapest');
    const comfortable = expectDefined(result.recommendations[1], 'comfortable');
    if (cheapest.pick.kind !== 'gpu-platform') throw new Error('attendu gpu-platform');
    expect(cheapest.pick.platform.id).toBe('plat-1');
    expect(cheapest.pick.totalEur).toBe(1_820);
    if (comfortable.pick.kind !== 'gpu-platform') throw new Error('attendu gpu-platform');
    expect(comfortable.pick.platform.id).toBe('plat-2');
    expect(comfortable.pick.totalEur).toBe(3_320);
  });
});

// ---------------------------------------------------------------------------
// Multi-unités de station (répliques data-parallèles, plafond 4)
// ---------------------------------------------------------------------------

describe('recommendHardware — multi-unités de station (répliques data-parallèles)', () => {
  it('échelonne les unités quand la cible de débit n’est pas atteinte par une seule', () => {
    // Station 128 Go (bw 800, flops 3e14) avec B = 64 séquences :
    // tps(B) = 680e9 / (6e9 + 64 × 131072 × 576) ≈ 62,8 tok/s < 70 ;
    // tps(B/2) = 680e9 / (6e9 + 32 × 131072 × 576) ≈ 80,8 tok/s ≥ 70 → 2 unités.
    // Prix = 2 × 7000 = 14 000 €, BOM quantité 2.
    const result = recommendHardware(
      SPEC_10B,
      scenarioOf({ simultaneous: 64, minTpsPerUser: 70 }),
      STATION_ONLY,
    );
    expect(result.status).toBe('viable');
    const recommendation = expectDefined(result.recommendations[0], 'recommandation');
    const pick = recommendation.pick;
    if (pick.kind !== 'station') throw new Error('attendu station');
    expect(pick.unitCount).toBe(2);
    expect(pick.totalEur).toBe(14_000);
    // Le sizing affiché reste celui de l'unité simple (tps non multiplié).
    expect(pick.sizing.tpsPerUser).toBeCloseTo(62.78, 1);
    const bomLine = expectDefined(pick.bom[0], 'ligne BOM station');
    expect(bomLine.quantity).toBe(2);
    expect(bomLine.unitPriceEur).toBe(7_000);
    expect(bomLine.totalEur).toBe(14_000);
  });

  it('plafonne le nombre d’unités à 4 quand la cible de débit l’exige (90 tok/s)', () => {
    // tps(B/4) = 680e9 / (6e9 + 16 × 131072 × 576) ≈ 94,3 ≥ 90 → 4 unités
    // (exactement le plafond) → 4 × 7000 = 28 000 €.
    const result = recommendHardware(
      SPEC_10B,
      scenarioOf({ simultaneous: 64, minTpsPerUser: 90 }),
      STATION_ONLY,
    );
    expect(result.status).toBe('viable');
    const pick = expectDefined(result.recommendations[0], 'recommandation').pick;
    if (pick.kind !== 'station') throw new Error('attendu station');
    expect(pick.unitCount).toBe(4);
    expect(pick.totalEur).toBe(28_000);
    expect(expectDefined(pick.bom[0], 'ligne BOM station').quantity).toBe(4);
  });

  it('rejette la station quand même 4 unités n’atteignent pas la cible débit (95 tok/s)', () => {
    // tps(B/4) ≈ 94,3 < 95 : aucune quantité ≤ plafond n'atteint la cible
    // dure → rejet explicite « Débit ».
    const result = recommendHardware(
      SPEC_10B,
      scenarioOf({ simultaneous: 64, minTpsPerUser: 95 }),
      STATION_ONLY,
    );
    expect(result.status).toBe('aucune');
    expect(result.reasons.some((r) => r.includes('Débit'))).toBe(true);
  });

  it('échelonne les unités selon la charge offerte et plafonne à 4', () => {
    // 1000 utilisateurs × 640 tokens/min = 10 666,7 tok/s de charge offerte
    // vs agrégat d'une unité ≈ 431,6 tok/s → 25 unités requises → plafond 4.
    const result = recommendHardware(
      SPEC_10B,
      scenarioOf({ totalUsers: 1000 }),
      STATION_ONLY,
    );
    expect(result.status).toBe('viable');
    const pick = expectDefined(result.recommendations[0], 'recommandation').pick;
    if (pick.kind !== 'station') throw new Error('attendu station');
    expect(pick.unitCount).toBe(4);
    expect(pick.totalEur).toBe(28_000);
    expect(expectDefined(pick.bom[0], 'ligne BOM station').quantity).toBe(4);
  });

  it('échelonne les unités pour la cible TTFT (préfill agrégé) et rejette au-delà du plafond', () => {
    // ttft(1) = 2 × 1e10 × 512 / (3e14 × 0.4) ≈ 0,0853 s.
    // Cible 0,06 s → 2 unités (0,0427 s ≤ 0,06) ; cible 0,02 s → 5 unités
    // au-delà du plafond → rejet « TTFT ».
    const ok = recommendHardware(SPEC_10B, scenarioOf({ ttftTargetSec: 0.06 }), STATION_ONLY);
    expect(ok.status).toBe('viable');
    const pick = expectDefined(ok.recommendations[0], 'recommandation').pick;
    if (pick.kind !== 'station') throw new Error('attendu station');
    expect(pick.unitCount).toBe(2);
    expect(pick.totalEur).toBe(14_000);

    const ko = recommendHardware(SPEC_10B, scenarioOf({ ttftTargetSec: 0.02 }), STATION_ONLY);
    expect(ko.status).toBe('aucune');
    expect(ko.reasons.some((r) => r.includes('TTFT'))).toBe(true);
  });

  it('rejette la station quand une unité seule ne tient pas les poids (pas de striping mémoire)', () => {
    // 100 Md FP16 : poids ≈ 186,3 Gio → VRAM requise ≈ 228 Gio > 119 Gio
    // (128 × 0,93) → gpuCount 2 sur l'unité simple → rejet VRAM, le
    // multi-unité n'est pas du striping mémoire.
    const SPEC_100B_FP16: ModelSpec = { ...SPEC_10B, totalParams: 100_000_000_000 };
    const result = recommendHardware(SPEC_100B_FP16, scenarioOf({ quant: 'FP16' }), STATION_ONLY);
    expect(result.status).toBe('aucune');
    expect(result.reasons.some((r) => r.includes('VRAM'))).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Catalogue réel : DGX B200 / DGX B300 et multi-unités Spark / Mac Studio
// ---------------------------------------------------------------------------

describe('recommendHardware — catalogue réel (DGX B200 / DGX B300)', () => {
  it('recommande DGX B200 et B300 en unité simple quand la charge tient', () => {
    // 500 Md FP16 : VRAM requise ≈ 1122 Gio — aucune plateforme GPU du
    // catalogue (8 × H200 × 0,93 ≈ 1049 Gio) ne tient ; seules les stations
    // DGX B200 (1440 Gio) et DGX B300 (2304 Gio) passent.
    const result = recommendHardware(SPEC_500B_FP16, scenarioOf({ quant: 'FP16' }), REAL_CATALOG);
    expect(result.status).toBe('viable');
    expect(result.recommendations).toHaveLength(2);

    const cheapest = roleOf(result, 'cheapest');
    const comfortable = roleOf(result, 'comfortable');
    if (cheapest.kind !== 'station' || comfortable.kind !== 'station') {
      throw new Error('attendu des stations DGX B200/B300');
    }
    expect(cheapest.station.id).toBe('nvidia-dgx-b300');
    expect(cheapest.unitCount).toBe(1);
    expect(cheapest.totalEur).toBe(500_000);
    expect(comfortable.station.id).toBe('nvidia-dgx-b200');
    expect(comfortable.unitCount).toBe(1);
    expect(comfortable.totalEur).toBe(600_000);
  });

  it('multiplie les unités DGX Spark / Mac Studio quand une unité ne suffit pas (scénario golden)', () => {
    // Charge offerte 3750 tok/s (30 users × 3 req/min × 2500 tokens) :
    // Mac Studio (bw 819) : agrégat ≈ 1660,8 tok/s → 3 unités (19 797 €).
    // DGX Spark (bw 273) : agrégat ≈ 553,6 tok/s → 7 unités → plafond 4
    // (22 516 €). Le sizing affiché reste celui de l'unité simple.
    const result = recommendHardware(SPEC_MOE_30B, SCENARIO_GOLDEN, REAL_STATIONS_SPARK_MAC);
    expect(result.status).toBe('viable');

    const cheapest = roleOf(result, 'cheapest');
    const comfortable = roleOf(result, 'comfortable');
    if (cheapest.kind !== 'station' || comfortable.kind !== 'station') {
      throw new Error('attendu des stations Spark/Mac Studio');
    }
    expect(cheapest.station.id).toBe('mac-studio-m5-ultra-96');
    expect(cheapest.unitCount).toBe(3);
    expect(cheapest.totalEur).toBe(19_797);
    expect(expectDefined(cheapest.bom[0], 'ligne BOM station').quantity).toBe(3);
    expect(comfortable.station.id).toBe('nvidia-dgx-spark');
    expect(comfortable.unitCount).toBe(4);
    expect(comfortable.totalEur).toBe(22_516);
    expect(expectDefined(comfortable.bom[0], 'ligne BOM station').quantity).toBe(4);
    // Sizing par unité inchangé : tps et agrégat de l'unité simple.
    expect(cheapest.sizing.tpsPerUser).toBeCloseTo(166.08, 1);
    expect(comfortable.sizing.aggregateToksPerSec).toBeCloseTo(553.6, 0);
  });
});
