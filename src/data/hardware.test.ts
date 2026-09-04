/**
 * Tests du catalogue matériel (étape 4 du plan).
 *
 * Le JSON est importé en tant que chaîne via `?raw` (tsconfig sans
 * resolveJsonModule) puis parsé et validé : le fichier de données doit rester
 * conforme au schéma et aux invariants métier du design (§5.1-§5.4).
 */
import { describe, expect, it } from 'vitest';
import raw from './hardware.json?raw';
import { validateHardwareCatalog } from './hardware.schema';
import type {
  CatalogGpu,
  CatalogPlatform,
  CatalogStation,
  HardwareCatalog,
  HardwareItem,
  RamKit,
} from './hardware.schema';

const catalog: HardwareCatalog = JSON.parse(raw);

const items: readonly HardwareItem[] = catalog.items;
const gpuList = items.filter((item): item is CatalogGpu => item.category === 'gpu');
const stationList = items.filter((item): item is CatalogStation => item.category === 'station');
const platformList = items.filter((item): item is CatalogPlatform => item.category === 'platform');
const ramList = items.filter((item): item is RamKit => item.category === 'ram');
const idSet = new Set(items.map((item) => item.id));

/** GPU exigés par le design §5.1 / plan (10 modèles minimum). */
const REQUIRED_GPU_IDS = [
  'rtx-4080-super',
  'rtx-4090',
  'rtx-5080',
  'rtx-5090',
  'rtx-6000-ada-48go',
  'rtx-pro-6000-blackwell-96go',
  'l40s-48go',
  'a100-80go-pcie',
  'h100-80go-pcie',
  'h200-141go',
] as const;

const REQUIRED_STATION_IDS = [
  'nvidia-dgx-spark',
  'gmktec-evo-x2',
  'framework-desktop',
  'mac-studio-m5-ultra-96',
] as const;

const REQUIRED_PLATFORM_IDS = ['desktop-2gpu', 'workstation-4gpu', 'rack-8gpu'] as const;

describe('Catalogue matériel (src/data/hardware.json)', () => {
  it('passe la validation complète du schéma', () => {
    expect(validateHardwareCatalog(catalog)).toEqual([]);
  });

  it('couvre les GPU du design §5.1 (grand public, pro, datacenter)', () => {
    for (const id of REQUIRED_GPU_IDS) {
      expect(idSet.has(id), `GPU requis manquant : ${id}`).toBe(true);
    }
  });

  it('couvre les stations compactes (DGX Spark, Strix Halo ×2, Mac Studio Ultra)', () => {
    expect(stationList.length).toBeGreaterThanOrEqual(4);
    for (const id of REQUIRED_STATION_IDS) {
      expect(idSet.has(id), `Station requise manquante : ${id}`).toBe(true);
    }
    const spark = stationList.find((station) => station.id === 'nvidia-dgx-spark');
    expect(spark?.bwGbps).toBeCloseTo(273, 0);
    const strix = stationList.find((station) => station.id === 'gmktec-evo-x2');
    expect(strix?.bwGbps).toBeCloseTo(256, 0);
  });

  it('chaque GPU applique le ratio utilisable ~93 % (design §5.1)', () => {
    for (const gpu of gpuList) {
      expect(Math.abs(gpu.usableVramRatio - 0.93)).toBeLessThanOrEqual(0.01);
    }
  });

  it('contient les 3 plateformes du design §5.3 (slots 2/4/8, prix croissants)', () => {
    expect(platformList.length).toBe(3);
    for (const id of REQUIRED_PLATFORM_IDS) {
      expect(idSet.has(id), `Plateforme requise manquante : ${id}`).toBe(true);
    }
    const slots = platformList.map((platform) => platform.maxGpuSlots).sort((a, b) => a - b);
    expect(slots).toEqual([2, 4, 8]);
    const prices = platformList.map((platform) => platform.priceEur).sort((a, b) => a - b);
    expect(prices.length).toBe(3);
    const [low, mid, high] = prices;
    expect(low ?? -1).toBeLessThan(mid ?? 0);
    expect(mid ?? 0).toBeLessThan(high ?? 0);
  });

  it('couvre la RAM 64/128/256 Go à un prix de crise 2026 (12-18 €/Go)', () => {
    const capacities = ramList.map((kit) => kit.capacityGib).sort((a, b) => a - b);
    expect(capacities).toEqual([64, 128, 256]);
    for (const kit of ramList) {
      expect(kit.pricePerGibEur, kit.id).toBeGreaterThan(10);
      expect(kit.pricePerGibEur, kit.id).toBeLessThan(20);
    }
  });

  it('chaque entrée est datée du 04/09/2026, sourcée en https et marquée listed|estimate', () => {
    for (const item of items) {
      expect(item.priceDate, item.id).toBe('2026-09-04');
      expect(item.source.startsWith('https://'), item.id).toBe(true);
      expect(['listed', 'estimate'], item.id).toContain(item.priceQuality);
    }
  });

  it('les prix par segment respectent la hiérarchie du marché', () => {
    for (const gpu of gpuList) {
      if (gpu.segment === 'consumer') expect(gpu.priceEur, gpu.id).toBeLessThanOrEqual(5100);
      if (gpu.segment === 'pro') expect(gpu.priceEur, gpu.id).toBeGreaterThanOrEqual(7500);
      if (gpu.segment === 'datacenter') expect(gpu.priceEur, gpu.id).toBeGreaterThanOrEqual(27000);
    }
  });

  it('les ids sont uniques et en kebab-case', () => {
    expect(idSet.size).toBe(items.length);
    for (const item of items) {
      expect(item.id).toMatch(/^[a-z0-9-]+$/);
    }
  });
});
