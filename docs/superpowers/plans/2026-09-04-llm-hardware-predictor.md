# Prédicteur de dimensionnement matériel LLM — Plan d'implémentation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal :** Un site statique (React+TS, GitHub Pages, UI française) qui, à partir d'un modèle Hugging Face + quantification + profil de charge, calcule le matériel à acheter avec prix EUR.

**Architecture :** SPA Vite+React+TypeScript, 100 % client-side. Le moteur de calcul est un module TS pur isolé de React (`src/engine/`), la seule partie testée en profondeur (vitest). L'API publique Hugging Face est appelée depuis le navigateur (sans clé) ; le catalogue matériel est un JSON versionné dans le repo.

**Tech Stack :** Vite 7, React 19, TypeScript 5.9 (strict), vitest 3 + @testing-library/react (smoke UI), GitHub Actions → Pages. Zéro dépendance runtime autre que react/react-dom.

**Spec de référence :** `docs/superpowers/specs/2026-09-04-llm-hardware-predictor-design.md` (approuvé).

## Global Constraints

- UI **entièrement en français** (labels, messages, erreurs, README).
- **Aucun backend, aucune clé API** : uniquement des fetch vers l'API publique HF (CORS ok).
- Moteur de calcul **pur** (aucun import React dans `src/engine/`), testable sans navigateur.
- Quantifications et bpw **exacts** : FP16=16, Q8_0=8.5, Q6_K=6.6, Q5_K_M=5.7, Q4_K_M=4.8.
- Constantes moteur : overhead 2 Gio fixes, `BW_EFFICIENCY=0.85`, `PREFILL_ETA=0.4`, RAM `max(64 Go, 1.5 × VRAM physique)`, **max 8 GPU**.
- Prix catalogue : EUR, indicatifs, champ `priceDate` affiché (« prix relevés du 2026-09 »).
- Bandeau permanent dans l'UI : « Estimations ±25-30 % — KV en FP16, continuous batching, interconnexion non modélisée ».
- TDD : test d'abord, il doit échouer, implémentation minimale, test vert, **commit à chaque tâche**.
- Scripts npm : `npm test` = `vitest run` (non interactif, pour CI).
- Ne jamais supprimer/fausser un test pour le faire passer ; si un test contredit la formule, on corrige le test avec une justification écrite (cf. note Task 5).

## Structure de fichiers (verrouillée)

```
local-llm-infra/
├── index.html                       # entry Vite, lang="fr"
├── package.json / tsconfig.json / vite.config.ts
├── .gitignore
├── .github/workflows/deploy.yml     # Pages (tâche 13)
├── README.md                        # (tâche 13)
├── docs/superpowers/specs/…         # existant (commité)
└── src/
    ├── main.tsx                     # bootstrap React
    ├── App.tsx                      # layout 2 colonnes + état global
    ├── App.css
    ├── types.ts                     # types partagés + QUANTIZATIONS (bpw)
    ├── engine/                      # TS pur, zéro React
    │   ├── model.ts                 # résolution HF → ModelSpec (+cache, 429 retry)
    │   ├── memory.ts                # poids, KV, VRAM+marge, ceil GPU, RAM
    │   ├── performance.ts           # roofline décode, TTFT, charge offerte
    │   ├── recommend.ts             # énumération/filtrage/classification picks + BOM
    │   └── index.ts                 # computeSizing() orchestrateur
    ├── data/hardware.json           # catalogue versionné (prix EUR)
    └── components/
        ├── ModelSearch.tsx          # autocomplete HF + gestion erreurs
        ├── ScenarioForm.tsx         # quantification + charge + cibles
        ├── ResultsPanel.tsx         # camembert, perf ✓/✗, picks, BOM, bandeau
        └── DonutChart.tsx           # SVG camembert 3 segments
```

Tests : `src/engine/*.test.ts` à côté des modules ; `src/App.test.tsx`, `src/components/*.test.tsx` (smoke).

---

### Task 1 : Scaffold Vite + React + TS + vitest

**Files:**
- Create: `package.json`, `tsconfig.json`, `vite.config.ts`, `index.html`, `.gitignore`, `src/main.tsx`, `src/App.tsx`, `src/App.css`, `src/vite-env.d.ts`, `src/App.test.tsx`

**Interfaces:**
- Consumes: rien (première tâche).
- Produces: projet compilable ; `App` (default export) affichant `Prédicteur LLM — dimensionnement matériel` ; scripts `npm test` / `npm run build` fonctionnels ; `package-lock.json` commité (requis pour `npm ci` en CI).

- [ ] **Step 1 : Créer les fichiers de config**

`package.json` :
```json
{
  "name": "llm-hardware-predictor",
  "private": true,
  "version": "0.1.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "tsc --noEmit && vite build",
    "test": "vitest run",
    "test:watch": "vitest"
  },
  "dependencies": {
    "react": "^19.1.0",
    "react-dom": "^19.1.0"
  },
  "devDependencies": {
    "@testing-library/dom": "^10.4.0",
    "@testing-library/react": "^16.3.0",
    "@types/react": "^19.1.0",
    "@types/react-dom": "^19.1.0",
    "@vitejs/plugin-react": "^4.6.0",
    "jsdom": "^25.0.0",
    "typescript": "~5.9.0",
    "vite": "^7.1.0",
    "vitest": "^3.2.0"
  }
}
```

`tsconfig.json` :
```json
{
  "compilerOptions": {
    "target": "ES2022",
    "lib": ["ES2022", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "skipLibCheck": true,
    "isolatedModules": true,
    "noEmit": true,
    "resolveJsonModule": true,
    "types": ["vite/client"]
  },
  "include": ["src"]
}
```

`vite.config.ts` :
```ts
/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: { environment: 'jsdom' },
});
```

`index.html` :
```html
<!doctype html>
<html lang="fr">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Prédicteur LLM — dimensionnement matériel</title>
  </head>
  <body>
    <div id="root"></div>
    <script type="module" src="/src/main.tsx"></script>
  </body>
</html>
```

`.gitignore` :
```
node_modules/
dist/
*.local
.DS_Store
```

`src/vite-env.d.ts` :
```ts
/// <reference types="vite/client" />
```

`src/main.tsx` :
```tsx
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './App.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
```

`src/App.tsx` (placeholder de scaffold — remplacé tâche 10) :
```tsx
export default function App() {
  return (
    <main>
      <h1>Prédicteur LLM — dimensionnement matériel</h1>
      <p>Scaffold OK — le formulaire arrive aux tâches suivantes.</p>
    </main>
  );
}
```

`src/App.css` :
```css
:root {
  color-scheme: light;
  font-family: system-ui, sans-serif;
}
body { margin: 0; }
main { max-width: 1100px; margin: 0 auto; padding: 1.5rem; }
```

- [ ] **Step 2 : Écrire le premier test (smoke)**

`src/App.test.tsx` :
```tsx
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import App from './App';

describe('App', () => {
  it('affiche le titre du site', () => {
    render(<App />);
    expect(screen.getByText('Prédicteur LLM — dimensionnement matériel')).toBeTruthy();
  });
});
```

- [ ] **Step 3 : Installer et vérifier**

```bash
npm install
npm test
npm run build
```
Attendu : 1 test PASS ; build sans erreur ; `dist/` créé ; `package-lock.json` généré.

- [ ] **Step 4 : Commit**

```bash
git add -A
git commit -m "chore: scaffold Vite + React 19 + TS strict + vitest"
```

---

### Task 2 : Types partagés + quantifications (types.ts)

**Files:**
- Create: `src/types.ts`
- Test: `src/types.test.ts`

**Interfaces:**
- Consumes: rien.
- Produces: `QuantizationId`, `QuantInfo`, `QUANTIZATIONS`, `bitsPerWeight(q): number`, `ModelSpec`, `Scenario`, `GpuEntry`, `StationEntry`, `PlatformEntry`, `HardwareCatalog`, `BomLine`, `MemoryEstimate`, `PerfEstimate`, `HardwarePick`, `SizingResult`. Toutes les tâches suivantes importent depuis `../types` (ou `./types`).

- [ ] **Step 1 : Test échouant**

`src/types.test.ts` :
```ts
import { describe, expect, it } from 'vitest';
import { bitsPerWeight, QUANTIZATIONS } from './types';

describe('QUANTIZATIONS', () => {
  it('expose les 5 quantifications avec bpw exacts', () => {
    expect(QUANTIZATIONS.map((q) => q.id)).toEqual(['FP16', 'Q8_0', 'Q6_K', 'Q5_K_M', 'Q4_K_M']);
    expect(bitsPerWeight('FP16')).toBe(16);
    expect(bitsPerWeight('Q8_0')).toBe(8.5);
    expect(bitsPerWeight('Q6_K')).toBe(6.6);
    expect(bitsPerWeight('Q5_K_M')).toBe(5.7);
    expect(bitsPerWeight('Q4_K_M')).toBe(4.8);
  });

  it('rejette une quantification inconnue', () => {
    expect(() => bitsPerWeight('Q2' as never)).toThrow(/inconnue/);
  });
});
```

- [ ] **Step 2 : Exécuter — doit échouer**

```bash
npm test -- types
```
Attendu : FAIL (module `./types` introuvable).

- [ ] **Step 3 : Implémentation**

`src/types.ts` :
```ts
export type QuantizationId = 'FP16' | 'Q8_0' | 'Q6_K' | 'Q5_K_M' | 'Q4_K_M';

export interface QuantInfo {
  id: QuantizationId;
  label: string;
  bpw: number;
}

export const QUANTIZATIONS: readonly QuantInfo[] = [
  { id: 'FP16', label: 'FP16 (16 bits)', bpw: 16 },
  { id: 'Q8_0', label: 'Q8_0 (~8,5 bits/poids)', bpw: 8.5 },
  { id: 'Q6_K', label: 'Q6_K (~6,6 bits/poids)', bpw: 6.6 },
  { id: 'Q5_K_M', label: 'Q5_K_M (~5,7 bits/poids)', bpw: 5.7 },
  { id: 'Q4_K_M', label: 'Q4_K_M (~4,8 bits/poids)', bpw: 4.8 },
];

export function bitsPerWeight(q: QuantizationId): number {
  const info = QUANTIZATIONS.find((x) => x.id === q);
  if (!info) throw new Error(`Quantification inconnue : ${q}`);
  return info.bpw;
}

export interface ModelSpec {
  id: string;
  name: string;
  /** N — paramètres totaux (safetensors.total), tous experts MoE inclus. */
  paramsTotal: number;
  /** A — paramètres actifs par token (≈ paramsTotal si dense). */
  paramsActive: number;
  isMoe: boolean;
  layers: number;
  kvHeads: number;
  headDim: number;
  hiddenSize: number;
  /** Champs déduits par fallback → badges « valeurs déduites » dans l'UI. */
  derived: { kvHeads: boolean; headDim: boolean; moe: boolean };
}

export interface Scenario {
  concurrentUsers: number;
  totalUsers: number;
  requestsPerUserPerMin: number;
  avgInputTokens: number;
  avgOutputTokens: number;
  maxContextTokens: number;
  ttftTargetSec: number;
  minTpsPerUser: number;
  /** 0-100, défaut 20. */
  safetyMarginPct: number;
}

export interface GpuEntry {
  id: string;
  name: string;
  vramGB: number;
  usableVramGB: number;
  bandwidthGBps: number;
  flopsFP16: number;
  tdpW: number;
  priceEur: number;
}

export interface StationEntry {
  id: string;
  name: string;
  memoryGB: number;
  bandwidthGBps: number;
  flopsFP16: number;
  priceEur: number;
}

export interface PlatformEntry {
  id: string;
  name: string;
  gpuSlots: number;
  priceEur: number;
}

export interface HardwareCatalog {
  gpus: GpuEntry[];
  stations: StationEntry[];
  platforms: PlatformEntry[];
  ramPricePerGB: number;
  priceDate: string; // ex. "2026-09"
}

export interface BomLine {
  label: string;
  unitPriceEur: number;
  qty: number;
  lineTotalEur: number;
}

export interface MemoryEstimate {
  weightsBytes: number;
  kvBytes: number;
  overheadBytes: number;
  /** Poids + KV + overhead, marge de sécurité appliquée. */
  vramRequiredBytes: number;
}

export interface PerfEstimate {
  ttftSec: number;
  tpsPerUser: number;
  aggregateTps: number;
  offeredTps: number;
  meetsTtft: boolean;
  meetsTps: boolean;
  absorbsLoad: boolean;
}

export interface HardwarePick {
  label: string;
  category: 'la moins chère' | 'équilibrée' | 'confortable';
  gpuCount: number; // 0 pour une station
  gpuId?: string;
  platformId?: string;
  stationId?: string;
  ramGB: number;
  totalPriceEur: number;
  bom: BomLine[];
  memory: MemoryEstimate;
  perf: PerfEstimate;
  vramHeadroomPct: number;
}

export interface SizingResult {
  memory: MemoryEstimate;
  offeredTps: number;
  picks: HardwarePick[];
  maxGpuCountExceeded: boolean;
}
```

- [ ] **Step 4 : Exécuter — doit passer**

```bash
npm test -- types
```
Attendu : PASS.

- [ ] **Step 5 : Commit**

```bash
git add src/types.ts src/types.test.ts
git commit -m "feat: types partagés + table de quantifications (bpw GGUF)"
```

---

### Task 3 : Résolution de modèle HF (engine/model.ts)

**Files:**
- Create: `src/engine/model.ts`
- Test: `src/engine/model.test.ts`

**Interfaces:**
- Consumes: `ModelSpec` (types.ts).
- Produces:
  - `class ModelResolutionError extends Error`
  - `function buildSpec(id: string, name: string, paramsTotal: number, cfg: Record<string, unknown>): ModelSpec`
  - `async function searchModels(query: string): Promise<{ id: string }[]>`
  - `async function resolveModel(id: string): Promise<ModelSpec>` (cache mémoire par ID + 1 retry après 2 s sur HTTP 429)

- [ ] **Step 1 : Tests échouants**

`src/engine/model.test.ts` :
```ts
import { describe, expect, it, vi } from 'vitest';
import { buildSpec, ModelResolutionError, resolveModel, searchModels } from './model';

const QWEN30B_CFG = {
  num_hidden_layers: 48,
  num_attention_heads: 32,
  num_key_value_heads: 4,
  head_dim: 128,
  hidden_size: 2048,
  intermediate_size: 768,
  num_experts: 128,
  num_experts_per_tok: 8,
  vocab_size: 151936,
};

describe('buildSpec', () => {
  const N = 30_500_000_000;

  it('MoE : A ≈ attention + embed + experts actifs (Qwen3-30B-A3B)', () => {
    const spec = buildSpec('Qwen/Qwen3-30B-A3B', 'Qwen3 30B A3B', N, QWEN30B_CFG);
    expect(spec.isMoe).toBe(true);
    // attn = 48×2048×128×(2×32+2×4) ≈ 0,906 Md ; mlp/expert = 48×3×768×2048 ≈ 0,226 Md
    // embed = 151936×2048 ≈ 0,311 Md ; A ≈ 0,906 + 8×0,226 + 0,311 ≈ 3,03 Md
    expect(spec.paramsActive).toBeGreaterThan(2.9e9);
    expect(spec.paramsActive).toBeLessThan(3.2e9);
    expect(spec.paramsTotal).toBe(N);
    expect(spec.derived.kvHeads).toBe(false);
  });

  it('dense sans fallback : aucun badge déduit', () => {
    const cfg = { num_hidden_layers: 32, num_attention_heads: 32, num_key_value_heads: 8, head_dim: 128, hidden_size: 4096 };
    const spec = buildSpec('dense/x', 'Dense 8B', 8_030_000_000, cfg);
    expect(spec.isMoe).toBe(false);
    expect(spec.paramsActive).toBe(8_030_000_000);
    expect(spec.derived).toEqual({ kvHeads: false, headDim: false, moe: false });
  });

  it('fallbacks : kvHeads ← attention heads, headDim ← hidden/heads, badges déduits', () => {
    const cfg = { num_hidden_layers: 32, num_attention_heads: 32, hidden_size: 4096 };
    const spec = buildSpec('fb/x', 'FB', 1e9, cfg);
    expect(spec.kvHeads).toBe(32);
    expect(spec.headDim).toBe(128);
    expect(spec.derived.kvHeads).toBe(true);
    expect(spec.derived.headDim).toBe(true);
  });

  it('MoE sans intermediate_size : A ≈ N × perTok/experts + badge moe', () => {
    const cfg = { ...QWEN30B_CFG, intermediate_size: undefined };
    const spec = buildSpec('moe/x', 'MoE grossier', 30e9, cfg);
    expect(spec.paramsActive).toBeCloseTo(30e9 * (8 / 128), -6);
    expect(spec.derived.moe).toBe(true);
  });
});

describe('resolveModel / searchModels (fetch mocké)', () => {
  it('searchModels : <3 caractères → pas d’appel réseau', async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal('fetch', fetchMock);
    expect(await searchModels('ab')).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
    vi.unstubAllGlobals();
  });

  it('resolveModel : 404 → ModelResolutionError « introuvable »', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{}', { status: 404 })));
    await expect(resolveModel('a/b')).rejects.toThrow(ModelResolutionError);
    vi.unstubAllGlobals();
  });

  it('resolveModel : sans safetensors → erreur « pointe le modèle de base »', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({}), { status: 200 })));
    await expect(resolveModel('a/b-GGUF')).rejects.toThrow(/safetensors/);
    vi.unstubAllGlobals();
  });

  it('resolveModel : happy path → ModelSpec mis en cache (2e appel sans fetch)', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce(new Response(JSON.stringify({ modelName: 'X', safetensors: { total: 1e9 } }), { status: 200 }))
      .mockResolvedValueOnce(new Response(JSON.stringify(QWEN30B_CFG), { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);
    const spec1 = await resolveModel('org/x');
    expect(spec1.layers).toBe(48);
    const spec2 = await resolveModel('org/x');
    expect(spec2).toBe(spec1);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    vi.unstubAllGlobals();
  });
});
```

- [ ] **Step 2 : Exécuter — doit échouer**

```bash
npm test -- model
```
Attendu : FAIL (`./model` introuvable).

- [ ] **Step 3 : Implémentation**

`src/engine/model.ts` :
```ts
import type { ModelSpec } from '../types';

export class ModelResolutionError extends Error {}

const HF = 'https://huggingface.co';

export async function searchModels(query: string): Promise<{ id: string }[]> {
  const q = query.trim();
  if (q.length < 3) return [];
  const res = await fetch(
    `${HF}/api/models?search=${encodeURIComponent(q)}&pipeline_tag=text-generation&limit=10`,
  );
  if (!res.ok) throw new Error('Hugging Face injoignable — vérifie ta connexion');
  const hits: { id: string }[] = await res.json();
  return hits.map((m) => ({ id: m.id }));
}

async function fetchWithRetry(url: string): Promise<Response> {
  const first = await fetch(url);
  if (first.status === 429) {
    await new Promise((r) => setTimeout(r, 2000));
    const second = await fetch(url);
    if (second.status === 429) throw new Error('Limite de requêtes HF atteinte — réessaie dans un instant');
    return second;
  }
  return first;
}

export function buildSpec(
  id: string,
  name: string,
  paramsTotal: number,
  cfg: Record<string, unknown>,
): ModelSpec {
  const num = (v: unknown, dflt = 0): number => (typeof v === 'number' ? v : dflt);
  const layers = num(cfg.num_hidden_layers);
  const attentionHeads = num(cfg.num_attention_heads);
  const hiddenSize = num(cfg.hidden_size);
  const hasKv = typeof cfg.num_key_value_heads === 'number';
  const kvHeads = hasKv ? (cfg.num_key_value_heads as number) : attentionHeads;
  const hasHeadDim = typeof cfg.head_dim === 'number';
  const headDim = hasHeadDim
    ? (cfg.head_dim as number)
    : attentionHeads > 0
      ? hiddenSize / attentionHeads
      : 0;
  const numExperts = num(cfg.num_experts);
  const perTok = num(cfg.num_experts_per_tok);
  const isMoe = numExperts > 1 && perTok > 0;
  let paramsActive = paramsTotal;
  let moeApprox = false;
  if (isMoe) {
    const intermediate = num(cfg.intermediate_size);
    if (intermediate > 0) {
      // Attention partagée + experts actifs + embeddings (tied comptés une fois)
      const attn = layers * hiddenSize * headDim * (2 * attentionHeads + 2 * kvHeads);
      const mlpPerExpert = layers * 3 * intermediate * hiddenSize;
      const embed = num(cfg.vocab_size) * hiddenSize;
      paramsActive = attn + perTok * mlpPerExpert + embed;
    } else {
      // Pas la structure par expert → estimation grossière proportionnelle
      paramsActive = paramsTotal * (perTok / numExperts);
      moeApprox = true;
    }
  }
  return {
    id,
    name,
    paramsTotal,
    paramsActive,
    isMoe,
    layers,
    kvHeads,
    headDim,
    hiddenSize,
    derived: { kvHeads: !hasKv, headDim: !hasHeadDim, moe: moeApprox },
  };
}

const cache = new Map<string, ModelSpec>();

export async function resolveModel(id: string): Promise<ModelSpec> {
  const key = id.trim();
  const cached = cache.get(key);
  if (cached) return cached;
  const metaRes = await fetchWithRetry(`${HF}/api/models/${key}`);
  if (metaRes.status === 404) throw new ModelResolutionError(`Modèle introuvable : ${key}`);
  if (!metaRes.ok) throw new Error('Hugging Face injoignable — vérifie ta connexion');
  const meta: { modelName?: string; safetensors?: { total?: number } } = await metaRes.json();
  const total = meta.safetensors?.total;
  if (typeof total !== 'number') {
    throw new ModelResolutionError(
      'Impossible de déterminer les paramètres (pas de safetensors) — pointe le modèle de base',
    );
  }
  const cfgRes = await fetchWithRetry(`${HF}/${key}/raw/main/config.json`);
  if (!cfgRes.ok) throw new ModelResolutionError('config.json introuvable pour ce modèle');
  const cfg: Record<string, unknown> = await cfgRes.json();
  const spec = buildSpec(key, meta.modelName ?? key, total, cfg);
  cache.set(key, spec);
  return spec;
}
```

- [ ] **Step 4 : Exécuter — doit passer**

```bash
npm test -- model
```
Attendu : 8 PASS.

- [ ] **Step 5 : Commit**

```bash
git add src/engine/model.ts src/engine/model.test.ts
git commit -m "feat(engine): résolution HF → ModelSpec (MoE, fallbacks, cache, retry 429)"
```

---

### Task 4 : Mémoire — poids, KV cache, VRAM, GPU, RAM (engine/memory.ts)

**Files:**
- Create: `src/engine/memory.ts`
- Test: `src/engine/memory.test.ts`

**Interfaces:**
- Consumes: `ModelSpec`, `Scenario`, `QuantizationId`, `MemoryEstimate`, `bitsPerWeight` (types.ts).
- Produces:
  - `const OVERHEAD_BYTES: number` (2 Gio)
  - `const MAX_GPUS: number` (8)
  - `function weightsBytes(paramsTotal: number, q: QuantizationId): number`
  - `function kvBytesPerToken(spec: ModelSpec): number`
  - `function kvCacheBytes(spec: ModelSpec, scenario: Scenario): number`
  - `function estimateMemory(spec: ModelSpec, scenario: Scenario, q: QuantizationId): MemoryEstimate`
  - `function gpuCountFor(vramRequiredBytes: number, usableVramGB: number): number`
  - `function systemRamGB(totalVramGB: number): number`

- [ ] **Step 1 : Tests échouants**

`src/engine/memory.test.ts` :
```ts
import { describe, expect, it } from 'vitest';
import { estimateMemory, gpuCountFor, kvBytesPerToken, kvCacheBytes, MAX_GPUS, OVERHEAD_BYTES, systemRamGB, weightsBytes } from './memory';
import type { ModelSpec, Scenario } from '../types';

const QWEN30B: ModelSpec = {
  id: 'Qwen/Qwen3-30B-A3B', name: 'Qwen3 30B A3B', paramsTotal: 30_500_000_000,
  paramsActive: 3_030_000_000, isMoe: true, layers: 48, kvHeads: 4, headDim: 128,
  hiddenSize: 2048, derived: { kvHeads: false, headDim: false, moe: false },
};

const SC: Scenario = {
  concurrentUsers: 10, totalUsers: 30, requestsPerUserPerMin: 3,
  avgInputTokens: 2000, avgOutputTokens: 500, maxContextTokens: 16384,
  ttftTargetSec: 2, minTpsPerUser: 20, safetyMarginPct: 20,
};

describe('weightsBytes', () => {
  it('Q4_K_M : N × 4.8/8', () => {
    expect(weightsBytes(30_500_000_000, 'Q4_K_M')).toBe(30_500_000_000 * 4.8 / 8); // 18,3 Go
  });
  it('FP16 : N × 2', () => {
    expect(weightsBytes(1e9, 'FP16')).toBe(2e9);
  });
});

describe('kvCacheBytes', () => {
  it('Qwen3-30B-A3B : 2×48×4×128×2 = 98 304 o/token', () => {
    expect(kvBytesPerToken(QWEN30B)).toBe(98304);
  });
  it('16K × 10 simultanés ≈ 16,1 Go', () => {
    expect(kvCacheBytes(QWEN30B, SC)).toBe(98304 * 16384 * 10);
  });
});

describe('estimateMemory (golden mémoire)', () => {
  it('VRAM = (poids + KV + 2 Gio) × 1,2 ≈ 43,86 Go', () => {
    const m = estimateMemory(QWEN30B, SC, 'Q4_K_M');
    expect(m.weightsBytes).toBeCloseTo(18.3e9, -6);
    expect(m.kvBytes).toBeCloseTo(16.106e9, -6);
    expect(m.overheadBytes).toBe(2 * 1024 ** 3);
    expect(m.vramRequiredBytes).toBeCloseTo(43.864e9, -6);
  });
});

describe('gpuCountFor', () => {
  it('2 GPU pour 43,86 Go sur RTX 5090 (29,8 Go utilisables)', () => {
    expect(gpuCountFor(43.864e9, 29.8)).toBe(2);
  });
  it('1 GPU si ça tient juste', () => {
    expect(gpuCountFor(29e9, 29.8)).toBe(1);
  });
});

describe('systemRamGB', () => {
  it('max(64, 1.5 × VRAM physique)', () => {
    expect(systemRamGB(16)).toBe(64);
    expect(systemRamGB(64)).toBe(96);
  });
});

describe('constantes', () => {
  it('overhead 2 Gio, max 8 GPU', () => {
    expect(OVERHEAD_BYTES).toBe(2 * 1024 ** 3);
    expect(MAX_GPUS).toBe(8);
  });
});
```

- [ ] **Step 2 : Exécuter — doit échouer**

```bash
npm test -- memory
```
Attendu : FAIL.

- [ ] **Step 3 : Implémentation**

`src/engine/memory.ts` :
```ts
import type { MemoryEstimate, ModelSpec, QuantizationId, Scenario } from '../types';
import { bitsPerWeight } from '../types';

/** Contexte CUDA, activations, fragmentation. */
export const OVERHEAD_BYTES = 2 * 1024 ** 3;
export const MAX_GPUS = 8;

export function weightsBytes(paramsTotal: number, q: QuantizationId): number {
  return (paramsTotal * bitsPerWeight(q)) / 8;
}

/** K + V, par token, par séquence, KV en FP16. */
export function kvBytesPerToken(spec: ModelSpec): number {
  return 2 * spec.layers * spec.kvHeads * spec.headDim * 2;
}

export function kvCacheBytes(spec: ModelSpec, scenario: Scenario): number {
  return kvBytesPerToken(spec) * scenario.maxContextTokens * scenario.concurrentUsers;
}

export function estimateMemory(spec: ModelSpec, scenario: Scenario, q: QuantizationId): MemoryEstimate {
  const weights = weightsBytes(spec.paramsTotal, q);
  const kv = kvCacheBytes(spec, scenario);
  const margin = 1 + scenario.safetyMarginPct / 100;
  return {
    weightsBytes: weights,
    kvBytes: kv,
    overheadBytes: OVERHEAD_BYTES,
    vramRequiredBytes: (weights + kv + OVERHEAD_BYTES) * margin,
  };
}

export function gpuCountFor(vramRequiredBytes: number, usableVramGB: number): number {
  return Math.ceil(vramRequiredBytes / (usableVramGB * 1024 ** 3));
}

/** RAM système pour l'OS, les checkpoints et l'offloading éventuel. */
export function systemRamGB(totalVramGB: number): number {
  return Math.max(64, Math.ceil(1.5 * totalVramGB));
}
```

- [ ] **Step 4 : Exécuter — doit passer**

```bash
npm test -- memory
```
Attendu : PASS.

- [ ] **Step 5 : Commit**

```bash
git add src/engine/memory.ts src/engine/memory.test.ts
git commit -m "feat(engine): poids, KV cache, VRAM+marge, ceil GPU, RAM système"
```

---

### Task 5 : Performance — roofline décode, TTFT, charge offerte (engine/performance.ts)

**Files:**
- Create: `src/engine/performance.ts`
- Test: `src/engine/performance.test.ts`

**Interfaces:**
- Consumes: `ModelSpec`, `Scenario`, `QuantizationId`, `PerfEstimate` (types.ts), `kvBytesPerToken` (memory.ts).
- Produces:
  - `const BW_EFFICIENCY = 0.85`, `const PREFILL_ETA = 0.4`
  - `interface ComputeNode { bandwidthGBps: number; flopsFP16: number }` (agrégat multi-GPU)
  - `function decodeTpsPerUser(spec: ModelSpec, scenario: Scenario, q: QuantizationId, node: ComputeNode): number`
  - `function ttftSeconds(spec: ModelSpec, scenario: Scenario, q: QuantizationId, node: ComputeNode): number`
  - `function offeredTps(scenario: Scenario): number`
  - `function estimatePerformance(spec, scenario, q, node): PerfEstimate`

- [ ] **Step 1 : Tests échouants**

`src/engine/performance.test.ts` :
```ts
import { describe, expect, it } from 'vitest';
import { BW_EFFICIENCY, decodeTpsPerUser, estimatePerformance, offeredTps, PREFILL_ETA, ttftSeconds } from './performance';
import type { ModelSpec, Scenario } from '../types';

const QWEN30B: ModelSpec = {
  id: 'Qwen/Qwen3-30B-A3B', name: 'Qwen3 30B A3B', paramsTotal: 30_500_000_000,
  paramsActive: 3_030_000_000, isMoe: true, layers: 48, kvHeads: 4, headDim: 128,
  hiddenSize: 2048, derived: { kvHeads: false, headDim: false, moe: false },
};
const SC: Scenario = {
  concurrentUsers: 10, totalUsers: 30, requestsPerUserPerMin: 3,
  avgInputTokens: 2000, avgOutputTokens: 500, maxContextTokens: 16384,
  ttftTargetSec: 2, minTpsPerUser: 20, safetyMarginPct: 20,
};
const TWO_5090 = { bandwidthGBps: 2 * 1792, flopsFP16: 2 * 419 };

describe('constantes', () => {
  it('BW_EFFICIENCY 0.85, PREFILL_ETA 0.4', () => {
    expect(BW_EFFICIENCY).toBe(0.85);
    expect(PREFILL_ETA).toBe(0.4);
  });
});

describe('decodeTpsPerUser (golden)', () => {
  it('2×RTX 5090, batch 10, Q4_K_M : ≈ 756 tok/s/utilisateur (roofline)', () => {
    const tps = decodeTpsPerUser(QWEN30B, SC, 'Q4_K_M', TWO_5090);
    // octets/pas = 3.03e9×0.6 + 10×98304×2250 = 1.818e9 + 2.21184e9 = 4.02984e9
    // BW_eff = 2×1792×0.85 = 3046.4 Go/s → 3046.4e9/4.02984e9 ≈ 756
    expect(tps).toBeCloseTo(756, -1);
  });
});

describe('ttftSeconds (golden)', () => {
  it('préfill 2000 tokens sur 2×5090 : ≈ 0.036 s', () => {
    expect(ttftSeconds(QWEN30B, SC, 'Q4_K_M', TWO_5090)).toBeCloseTo(0.0362, 3);
  });
});

describe('offeredTps', () => {
  it('30 users × 3 req/min × 2500 tok / 60 = 3750 tok/s', () => {
    expect(offeredTps(SC)).toBe(3750);
  });
});

describe('estimatePerformance', () => {
  it('toutes cibles ✓ sur 2×5090', () => {
    const p = estimatePerformance(QWEN30B, SC, 'Q4_K_M', TWO_5090);
    expect(p.meetsTtft).toBe(true);
    expect(p.meetsTps).toBe(true);
    expect(p.absorbsLoad).toBe(true); // agrégé ≈ 7560 ≥ 3750
    expect(p.aggregateTps).toBeCloseTo(7560, -1);
  });

  it('Strix Halo (256 Go/s, 50 TFLOPS) : meetsTps ✓ (≈54 ≥ 20) mais absorbsLoad ✗', () => {
    // BW_eff = 256×0.85 = 217.6 Go/s → 217.6e9/4.02984e9 ≈ 54 tok/s ≥ 20 → meetsTps = true
    // agrégé ≈ 540 < 3750 → absorbsLoad = false. Le test encode la formule, pas l'intuition.
    const p = estimatePerformance(QWEN30B, SC, 'Q4_K_M', { bandwidthGBps: 256, flopsFP16: 50 });
    expect(p.meetsTps).toBe(true);
    expect(p.absorbsLoad).toBe(false);
  });
});
```

- [ ] **Step 2 : Exécuter — doit échouer**

```bash
npm test -- performance
```
Attendu : FAIL.

- [ ] **Step 3 : Implémentation**

`src/engine/performance.ts` :
```ts
import type { ModelSpec, PerfEstimate, QuantizationId, Scenario } from '../types';
import { bitsPerWeight } from '../types';
import { kvBytesPerToken } from './memory';

/** Fraction de bande passante réellement exploitable en décode (batching continu). */
export const BW_EFFICIENCY = 0.85;
/** Efficacité FLOPS du préfill (kernels réels vs crête). */
export const PREFILL_ETA = 0.4;

export interface ComputeNode {
  bandwidthGBps: number; // agrégat multi-GPU
  flopsFP16: number;    // TFLOPS FP16 dense, agrégat
}

export function decodeTpsPerUser(
  spec: ModelSpec,
  scenario: Scenario,
  q: QuantizationId,
  node: ComputeNode,
): number {
  const weightsPerToken = (spec.paramsActive * bitsPerWeight(q)) / 8;
  const avgContext = scenario.avgInputTokens + scenario.avgOutputTokens / 2;
  const kvRead = scenario.concurrentUsers * kvBytesPerToken(spec) * avgContext;
  const bytesPerStep = weightsPerToken + kvRead;
  const bytesPerSec = node.bandwidthGBps * BW_EFFICIENCY * 1e9;
  return bytesPerSec / bytesPerStep;
}

export function ttftSeconds(
  spec: ModelSpec,
  scenario: Scenario,
  _q: QuantizationId,
  node: ComputeNode,
): number {
  const flops = 2 * spec.paramsActive * scenario.avgInputTokens;
  return flops / (node.flopsFP16 * 1e12 * PREFILL_ETA);
}

export function offeredTps(scenario: Scenario): number {
  return (scenario.totalUsers * scenario.requestsPerUserPerMin * (scenario.avgInputTokens + scenario.avgOutputTokens)) / 60;
}

export function estimatePerformance(
  spec: ModelSpec,
  scenario: Scenario,
  q: QuantizationId,
  node: ComputeNode,
): PerfEstimate {
  const ttft = ttftSeconds(spec, scenario, q, node);
  const tps = decodeTpsPerUser(spec, scenario, q, node);
  const aggregate = tps * scenario.concurrentUsers;
  const offered = offeredTps(scenario);
  return {
    ttftSec: ttft,
    tpsPerUser: tps,
    aggregateTps: aggregate,
    offeredTps: offered,
    meetsTtft: ttft <= scenario.ttftTargetSec,
    meetsTps: tps >= scenario.minTpsPerUser,
    absorbsLoad: aggregate >= offered,
  };
}
```

- [ ] **Step 4 : Exécuter — doit passer**

```bash
npm test -- performance
```
Attendu : PASS.

- [ ] **Step 5 : Commit**

```bash
git add src/engine/performance.ts src/engine/performance.test.ts
git commit -m "feat(engine): roofline décode, TTFT préfill, charge offerte vs capacité"
```

---

### Task 6 : Catalogue matériel (src/data/hardware.json)

**Files:**
- Create: `src/data/hardware.json`
- Test: `src/data/hardware.test.ts`

**Interfaces:**
- Consumes: `HardwareCatalog` (types.ts).
- Produces: `import catalog from '../data/hardware.json'` (cast `as unknown as HardwareCatalog`), conforme au schéma, avec `priceDate`.

- [ ] **Step 1 : Écrire le catalogue**

`src/data/hardware.json` (valeurs indicatives — **à vérifier/affiner par recherche web au Step 2**) :
```json
{
  "priceDate": "2026-09",
  "ramPricePerGB": 4.5,
  "gpus": [
    { "id": "rtx4080s", "name": "RTX 4080 Super", "vramGB": 16, "usableVramGB": 14.9, "bandwidthGBps": 1002, "flopsFP16": 238, "tdpW": 220, "priceEur": 1099 },
    { "id": "rtx4090", "name": "RTX 4090", "vramGB": 24, "usableVramGB": 22.3, "bandwidthGBps": 1008, "flopsFP16": 330, "tdpW": 450, "priceEur": 1899 },
    { "id": "rtx5080", "name": "RTX 5080", "vramGB": 16, "usableVramGB": 14.9, "bandwidthGBps": 960, "flopsFP16": 240, "tdpW": 360, "priceEur": 1249 },
    { "id": "rtx5090", "name": "RTX 5090", "vramGB": 32, "usableVramGB": 29.8, "bandwidthGBps": 1792, "flopsFP16": 419, "tdpW": 575, "priceEur": 2799 },
    { "id": "rtx6000ada", "name": "RTX 6000 Ada", "vramGB": 48, "usableVramGB": 44.6, "bandwidthGBps": 960, "flopsFP16": 291, "tdpW": 300, "priceEur": 8499 },
    { "id": "rtxpro6000", "name": "RTX PRO 6000 Blackwell", "vramGB": 96, "usableVramGB": 89.3, "bandwidthGBps": 1793, "flopsFP16": 470, "tdpW": 600, "priceEur": 9999 },
    { "id": "l40s", "name": "L40S", "vramGB": 48, "usableVramGB": 44.6, "bandwidthGBps": 864, "flopsFP16": 362, "tdpW": 350, "priceEur": 8999 },
    { "id": "a100-80", "name": "A100 80 Go (SXM)", "vramGB": 80, "usableVramGB": 74.4, "bandwidthGBps": 2039, "flopsFP16": 312, "tdpW": 400, "priceEur": 11999 },
    { "id": "h100-80", "name": "H100 80 Go (SXM)", "vramGB": 80, "usableVramGB": 74.4, "bandwidthGBps": 3350, "flopsFP16": 990, "tdpW": 700, "priceEur": 27999 },
    { "id": "h200-141", "name": "H200 141 Go (SXM)", "vramGB": 141, "usableVramGB": 131.1, "bandwidthGBps": 4800, "flopsFP16": 990, "tdpW": 700, "priceEur": 34999 }
  ],
  "stations": [
    { "id": "dgx-spark", "name": "NVIDIA DGX Spark (128 Go)", "memoryGB": 128, "bandwidthGBps": 273, "flopsFP16": 120, "priceEur": 4499 },
    { "id": "strix-halo-128", "name": "Mini-PC AMD Strix Halo — Ryzen AI Max+ 395, 128 Go (type Framework Desktop)", "memoryGB": 128, "bandwidthGBps": 256, "flopsFP16": 50, "priceEur": 2199 },
    { "id": "mac-studio-m3u-256", "name": "Mac Studio M3 Ultra (256 Go)", "memoryGB": 256, "bandwidthGBps": 819, "flopsFP16": 80, "priceEur": 8499 }
  ],
  "platforms": [
    { "id": "desktop-2", "name": "Desktop (i7-14700F, alimentation 2 000 W, 2 slots GPU)", "gpuSlots": 2, "priceEur": 2499 },
    { "id": "ws-4", "name": "Workstation (Threadripper PRO 7975WX, double alimentation, 4 slots GPU)", "gpuSlots": 4, "priceEur": 8999 },
    { "id": "rack-8", "name": "Serveur rack (2× EPYC 9354, châssis 8 GPU, alimentations redondantes)", "gpuSlots": 8, "priceEur": 24999 }
  ]
}
```

- [ ] **Step 2 : Vérifier les valeurs par recherche web**

Rechercher et corriger si divergent de plus de ~20 % : « RTX 5090 specs bandwidth TFLOPS price EUR », « DGX Spark 128GB price EUR », « Ryzen AI Max+ 395 Framework Desktop 128GB price », « H200 SXM 141GB bandwidth price », « RTX PRO 6000 Blackwell 96GB specs price ». Vérifier `vramGB`, `bandwidthGBps`, `flopsFP16` (FP16 dense, pas sparse), `tdpW`, `priceEur` (prix rue constaté). Les prix restent indicatifs — l'ordre de grandeur doit être juste, pas le centime.

- [ ] **Step 3 : Test de schéma**

`src/data/hardware.test.ts` :
```ts
import { describe, expect, it } from 'vitest';
import catalog from './hardware.json';
import type { HardwareCatalog } from '../types';

const c = catalog as unknown as HardwareCatalog;

describe('catalogue hardware.json', () => {
  it('a une date de prix et un prix RAM > 0', () => {
    expect(c.priceDate).toMatch(/^\d{4}-\d{2}$/);
    expect(c.ramPricePerGB).toBeGreaterThan(0);
  });
  it('≥ 9 GPU avec champs complets et cohérents', () => {
    expect(c.gpus.length).toBeGreaterThanOrEqual(9);
    for (const g of c.gpus) {
      expect(g.usableVramGB).toBeGreaterThan(0);
      expect(g.usableVramGB).toBeLessThanOrEqual(g.vramGB);
      expect(g.bandwidthGBps).toBeGreaterThan(0);
      expect(g.flopsFP16).toBeGreaterThan(0);
      expect(g.priceEur).toBeGreaterThan(0);
    }
  });
  it('3 stations compactes (DGX Spark, Strix Halo, Mac Studio)', () => {
    expect(c.stations.map((s) => s.id).sort()).toEqual(['dgx-spark', 'mac-studio-m3u-256', 'strix-halo-128']);
  });
  it('plateformes : 2, 4 et 8 slots', () => {
    expect(c.platforms.map((p) => p.gpuSlots).sort((a, b) => a - b)).toEqual([2, 4, 8]);
  });
});
```

- [ ] **Step 4 : Exécuter — doit passer**

```bash
npm test -- hardware
```
Attendu : PASS.

- [ ] **Step 5 : Commit**

```bash
git add src/data/hardware.json src/data/hardware.test.ts
git commit -m "feat(data): catalogue matériel versionné — GPUs, stations IA compactes, plateformes (prix EUR indicatifs 2026-09)"
```

---

### Task 7 : Recommandation matériel (engine/recommend.ts)

**Files:**
- Create: `src/engine/recommend.ts`
- Test: `src/engine/recommend.test.ts`

**Interfaces:**
- Consumes: `estimateMemory`, `gpuCountFor`, `MAX_GPUS`, `systemRamGB` (memory.ts) ; `estimatePerformance`, `ComputeNode` (performance.ts) ; `BomLine`, `HardwareCatalog`, `HardwarePick`, `ModelSpec`, `QuantizationId`, `Scenario` (types.ts).
- Produces:
  - `function recommendPicks(spec: ModelSpec, scenario: Scenario, q: QuantizationId, catalog: HardwareCatalog): { picks: HardwarePick[]; maxGpuCountExceeded: boolean }`

**Règles de classification (verrouillées) :** les picks viables (VRAM avec marge + cibles TTFT/débit respectées) sont triés par prix. « la moins chère » = 1er par prix ; « confortable » = headroom VRAM max (égalité → le moins cher) ; « équilibrée » = parmi les restants, celui dont le prix est le plus proche de la médiane des prix viables (uniquement s'il reste ≥ 3 viables). Un même pick ne peut pas porter deux catégories (dédoublonné). `absorbsLoad` est **informatif** (affiché ✓/✗) mais ne filtre pas.

- [ ] **Step 1 : Tests échouants**

`src/engine/recommend.test.ts` :
```ts
import { describe, expect, it } from 'vitest';
import { recommendPicks } from './recommend';
import type { HardwareCatalog, ModelSpec, Scenario } from '../types';

const SPEC: ModelSpec = {
  id: 'Qwen/Qwen3-30B-A3B', name: 'Qwen3 30B A3B', paramsTotal: 30_500_000_000,
  paramsActive: 3_030_000_000, isMoe: true, layers: 48, kvHeads: 4, headDim: 128,
  hiddenSize: 2048, derived: { kvHeads: false, headDim: false, moe: false },
};
const SC: Scenario = {
  concurrentUsers: 10, totalUsers: 30, requestsPerUserPerMin: 3,
  avgInputTokens: 2000, avgOutputTokens: 500, maxContextTokens: 16384,
  ttftTargetSec: 2, minTpsPerUser: 20, safetyMarginPct: 20,
};

const CATALOG: HardwareCatalog = {
  priceDate: '2026-09',
  ramPricePerGB: 4.5,
  gpus: [
    { id: 'small', name: 'Petit 16 Go', vramGB: 16, usableVramGB: 14.9, bandwidthGBps: 1000, flopsFP16: 240, tdpW: 220, priceEur: 1000 },
    { id: 'big', name: 'Gros 32 Go', vramGB: 32, usableVramGB: 29.8, bandwidthGBps: 1800, flopsFP16: 420, tdpW: 575, priceEur: 2800 },
  ],
  stations: [
    { id: 'mini', name: 'Station 128 Go lente', memoryGB: 128, bandwidthGBps: 256, flopsFP16: 50, priceEur: 2000 },
  ],
  platforms: [
    { id: 'desk', name: 'Desktop', gpuSlots: 2, priceEur: 500 },
    { id: 'rack', name: 'Rack', gpuSlots: 8, priceEur: 5000 },
  ],
};

describe('recommendPicks (VRAM requise ≈ 43,86 Go)', () => {
  const { picks, maxGpuCountExceeded } = recommendPicks(SPEC, SC, 'Q4_K_M', CATALOG);

  it('petit GPU : 3 GPU requis > 2 slots desktop → aucun pick small+desk', () => {
    expect(picks.find((p) => p.gpuId === 'small' && p.platformId === 'desk')).toBeUndefined();
  });

  it('station viable (tps/user ≈ 54 ≥ 20, TTFT 0,61 s ≤ 2 s) MAIS absorbsLoad ✗ — ne filtre pas', () => {
    const st = picks.find((p) => p.stationId === 'mini');
    expect(st).toBeDefined();
    expect(st!.perf.absorbsLoad).toBe(false);
  });

  it('la moins chère = la station (2 000 €), BOM à une ligne', () => {
    const cheapest = picks.find((p) => p.category === 'la moins chère');
    expect(cheapest?.stationId).toBe('mini');
    expect(cheapest?.totalPriceEur).toBe(2000);
    expect(cheapest?.bom).toEqual([
      { label: 'Station 128 Go lente', unitPriceEur: 2000, qty: 1, lineTotalEur: 2000 },
    ]);
  });

  it('équilibrée = 2× Gros sur desktop (2×2800 + 500 + RAM 96 Go × 4,5 = 6 532 €)', () => {
    // RAM : totalVramGB = 64 → max(64, 1,5×64) = 96 Go
    const balanced = picks.find((p) => p.category === 'équilibrée');
    expect(balanced?.gpuId).toBe('big');
    expect(balanced?.gpuCount).toBe(2);
    expect(balanced?.platformId).toBe('desk');
    expect(balanced?.totalPriceEur).toBe(2 * 2800 + 500 + 96 * 4.5);
  });

  it('≤ 3 picks, catégories uniques, ≤ 8 GPU, BOM = total', () => {
    expect(picks.length).toBeLessThanOrEqual(3);
    expect(new Set(picks.map((p) => p.category)).size).toBe(picks.length);
    for (const p of picks) {
      expect(p.gpuCount).toBeLessThanOrEqual(8);
      expect(p.bom.reduce((a, l) => a + l.lineTotalEur, 0)).toBe(p.totalPriceEur);
    }
  });

  it('confortable dédoublonné avec la moins chère (même pick) → absent', () => {
    // La station a le headroom max (68 %) ET le prix min → elle porte « la moins chère »
    // et « confortable » est dédoublonné : 2 picks au total ici.
    expect(picks.length).toBe(2);
    expect(picks.find((p) => p.category === 'confortable')).toBeUndefined();
    expect(maxGpuCountExceeded).toBe(false);
  });
});

describe('modèle énorme → maxGpuCountExceeded', () => {
  const HUGE: ModelSpec = { ...SPEC, paramsTotal: 500e9, paramsActive: 500e9 };
  const { picks, maxGpuCountExceeded } = recommendPicks(HUGE, SC, 'FP16', CATALOG);
  it('flag levé, aucun pick (stations trop petites, GPU > 8)', () => {
    expect(maxGpuCountExceeded).toBe(true);
    expect(picks.length).toBe(0);
  });
});

describe('charge légère → la station reste recommandée', () => {
  const SC1: Scenario = { ...SC, concurrentUsers: 1, totalUsers: 1, requestsPerUserPerMin: 1, maxContextTokens: 4096, ttftTargetSec: 5 };
  const { picks } = recommendPicks(SPEC, SC1, 'Q4_K_M', CATALOG);
  it('au moins un pick station apparaît', () => {
    expect(picks.some((p) => p.stationId === 'mini')).toBe(true);
  });
});
```

- [ ] **Step 2 : Exécuter — doit échouer**

```bash
npm test -- recommend
```
Attendu : FAIL.

- [ ] **Step 3 : Implémentation**

`src/engine/recommend.ts` :
```ts
import type {
  BomLine, HardwareCatalog, HardwarePick, ModelSpec, QuantizationId, Scenario,
} from '../types';
import { estimateMemory, gpuCountFor, MAX_GPUS, systemRamGB } from './memory';
import { estimatePerformance } from './performance';

const GIB = 1024 ** 3;

function bomTotal(bom: BomLine[]): number {
  return bom.reduce((acc, l) => acc + l.lineTotalEur, 0);
}

function headroomPct(capacityBytes: number, requiredBytes: number): number {
  return ((capacityBytes - requiredBytes) / capacityBytes) * 100;
}

export function recommendPicks(
  spec: ModelSpec,
  scenario: Scenario,
  q: QuantizationId,
  catalog: HardwareCatalog,
): { picks: HardwarePick[]; maxGpuCountExceeded: boolean } {
  const memory = estimateMemory(spec, scenario, q);
  const required = memory.vramRequiredBytes;
  const viable: HardwarePick[] = [];

  // Stations compactes (mémoire unifiée, compute intégré, pas de RAM additionnelle)
  for (const st of catalog.stations) {
    if (st.memoryGB * GIB < required) continue;
    const perf = estimatePerformance(spec, scenario, q, {
      bandwidthGBps: st.bandwidthGBps,
      flopsFP16: st.flopsFP16,
    });
    if (!perf.meetsTtft || !perf.meetsTps) continue;
    const bom: BomLine[] = [
      { label: st.name, unitPriceEur: st.priceEur, qty: 1, lineTotalEur: st.priceEur },
    ];
    viable.push({
      label: st.name,
      category: 'la moins chère', // corrigé par la classification ci-dessous
      gpuCount: 0,
      stationId: st.id,
      ramGB: 0,
      totalPriceEur: bomTotal(bom),
      bom,
      memory,
      perf,
      vramHeadroomPct: headroomPct(st.memoryGB * GIB, required),
    });
  }

  // Configurations GPU × plateforme (plateformes triées par prix → dedup garde la moins chère)
  let minCountOverCatalog = Number.POSITIVE_INFINITY;
  for (const gpu of catalog.gpus) {
    const count = gpuCountFor(required, gpu.usableVramGB);
    minCountOverCatalog = Math.min(minCountOverCatalog, count);
    if (count > MAX_GPUS) continue;
    const perf = estimatePerformance(spec, scenario, q, {
      bandwidthGBps: count * gpu.bandwidthGBps,
      flopsFP16: count * gpu.flopsFP16,
    });
    if (!perf.meetsTtft || !perf.meetsTps) continue;
    for (const pf of catalog.platforms
      .filter((p) => p.gpuSlots >= count)
      .sort((a, b) => a.priceEur - b.priceEur)) {
      const totalVramGB = count * gpu.vramGB;
      const ramGB = systemRamGB(totalVramGB);
      const bom: BomLine[] = [
        { label: gpu.name, unitPriceEur: gpu.priceEur, qty: count, lineTotalEur: count * gpu.priceEur },
        { label: pf.name, unitPriceEur: pf.priceEur, qty: 1, lineTotalEur: pf.priceEur },
        { label: `RAM ${ramGB} Go (DDR5)`, unitPriceEur: catalog.ramPricePerGB, qty: ramGB, lineTotalEur: ramGB * catalog.ramPricePerGB },
      ];
      viable.push({
        label: `${count}× ${gpu.name} (${pf.name})`,
        category: 'la moins chère',
        gpuCount: count,
        gpuId: gpu.id,
        platformId: pf.id,
        ramGB,
        totalPriceEur: bomTotal(bom),
        bom,
        memory,
        perf,
        vramHeadroomPct: headroomPct(totalVramGB * GIB, required),
      });
    }
  }

  // Dédoublonnage : une seule plateforme (la moins chère, déjà triée) par (gpuId, count)
  const seen = new Set<string>();
  const dedup = viable.filter((p) => {
    if (p.gpuCount > 0) {
      const key = `${p.gpuId}|${p.gpuCount}`;
      if (seen.has(key)) return false;
      seen.add(key);
    }
    return true;
  });

  // Classification : moins chère / équilibrée / confortable
  const byPrice = [...dedup].sort((a, b) => a.totalPriceEur - b.totalPriceEur);
  const cheapest = byPrice[0];
  const comfortable = [...byPrice].sort(
    (a, b) => b.vramHeadroomPct - a.vramHeadroomPct || a.totalPriceEur - b.totalPriceEur,
  )[0];
  const remaining = byPrice.filter((p) => p !== cheapest && p !== comfortable);
  let balanced: HardwarePick | undefined;
  if (remaining.length > 0 && byPrice.length >= 3) {
    const median = byPrice[Math.floor(byPrice.length / 2)].totalPriceEur;
    balanced = remaining.reduce((best, p) =>
      Math.abs(p.totalPriceEur - median) < Math.abs(best.totalPriceEur - median) ? p : best,
    );
  }

  const chosen: HardwarePick[] = [];
  for (const [pick, category] of [
    [cheapest, 'la moins chère'],
    [balanced, 'équilibrée'],
    [comfortable, 'confortable'],
  ] as const) {
    if (pick && !chosen.includes(pick)) {
      chosen.push({ ...pick, category });
    }
  }

  return {
    picks: chosen,
    maxGpuCountExceeded: minCountOverCatalog > MAX_GPUS,
  };
}
```

- [ ] **Step 4 : Exécuter — doit passer**

```bash
npm test -- recommend
```
Attendu : PASS. Vérification manuelle attendue : viable = [station 2 000 €, big×2/desk 6 532 €, small×3/rack 8 324 €] ; cheapest = station ; comfortable = station (headroom 68 %) → dédoublonné ; balanced = big×2/desk ; picks = 2 ; « charge légère » : station + GPU picks présents.

- [ ] **Step 5 : Commit**

```bash
git add src/engine/recommend.ts src/engine/recommend.test.ts
git commit -m "feat(engine): recommandation matériel — énumération, filtrage, moins chère/équilibrée/confortable, BOM"
```

---

### Task 8 : Orchestrateur computeSizing + golden bout-en-bout (engine/index.ts)

**Files:**
- Create: `src/engine/index.ts`
- Test: `src/engine/index.test.ts`

**Interfaces:**
- Consumes: `estimateMemory` (memory.ts), `offeredTps` (performance.ts), `recommendPicks` (recommend.ts).
- Produces: `function computeSizing(spec: ModelSpec, scenario: Scenario, q: QuantizationId, catalog: HardwareCatalog): SizingResult` — API unique consommée par l'UI. Re-exports : `resolveModel`, `searchModels`, `buildSpec`, `ModelResolutionError` (model), `estimateMemory`, `systemRamGB`, `weightsBytes`, `MAX_GPUS` (memory), `BW_EFFICIENCY`, `PREFILL_ETA` (performance), `recommendPicks` (recommend).

- [ ] **Step 1 : Test golden bout-en-bout (scénario de validation utilisateur)**

`src/engine/index.test.ts` :
```ts
import { describe, expect, it } from 'vitest';
import { computeSizing } from './index';
import type { HardwareCatalog, ModelSpec, Scenario } from '../types';

const QWEN30B: ModelSpec = {
  id: 'Qwen/Qwen3-30B-A3B', name: 'Qwen3 30B A3B', paramsTotal: 30_500_000_000,
  paramsActive: 3_030_000_000, isMoe: true, layers: 48, kvHeads: 4, headDim: 128,
  hiddenSize: 2048, derived: { kvHeads: false, headDim: false, moe: false },
};
const SC: Scenario = {
  concurrentUsers: 10, totalUsers: 30, requestsPerUserPerMin: 3,
  avgInputTokens: 2000, avgOutputTokens: 500, maxContextTokens: 16384,
  ttftTargetSec: 2, minTpsPerUser: 20, safetyMarginPct: 20,
};
const CATALOG: HardwareCatalog = {
  priceDate: '2026-09',
  ramPricePerGB: 4.5,
  gpus: [
    { id: 'rtx5090', name: 'RTX 5090', vramGB: 32, usableVramGB: 29.8, bandwidthGBps: 1792, flopsFP16: 419, tdpW: 575, priceEur: 2799 },
    { id: 'rtxpro6000', name: 'RTX PRO 6000 Blackwell', vramGB: 96, usableVramGB: 89.3, bandwidthGBps: 1793, flopsFP16: 470, tdpW: 600, priceEur: 9999 },
  ],
  stations: [
    { id: 'mac', name: 'Mac Studio M3 Ultra (256 Go)', memoryGB: 256, bandwidthGBps: 819, flopsFP16: 80, priceEur: 9999 },
  ],
  platforms: [
    { id: 'desk', name: 'Desktop', gpuSlots: 2, priceEur: 2499 },
    { id: 'rack', name: 'Rack', gpuSlots: 8, priceEur: 24999 },
  ],
};

describe('golden : Qwen3-30B-A3B Q4_K_M, 10 simultanés, 16K, 2000+500, TTFT<2s', () => {
  const r = computeSizing(QWEN30B, SC, 'Q4_K_M', CATALOG);

  it('VRAM requise ≈ 43,86 Go, charge offerte 3 750 tok/s', () => {
    expect(r.memory.vramRequiredBytes).toBeCloseTo(43.864e9, -6);
    expect(r.offeredTps).toBe(3750);
  });

  it('≥ 1 pick viable, chaque pick respecte TTFT et débit', () => {
    expect(r.picks.length).toBeGreaterThanOrEqual(1);
    for (const p of r.picks) {
      expect(p.perf.meetsTtft && p.perf.meetsTps).toBe(true);
    }
  });

  it('la moins chère = 2× RTX 5090 sur desktop (2×2799 + 2499 + 96×4,5 = 8 529 €)', () => {
    const cheapest = r.picks.find((p) => p.category === 'la moins chère');
    expect(cheapest?.gpuId).toBe('rtx5090');
    expect(cheapest?.gpuCount).toBe(2);
    expect(cheapest?.totalPriceEur).toBe(2 * 2799 + 2499 + 96 * 4.5);
  });

  it('confortable = Mac Studio (headroom 84 % > 57 % PRO > 36 % 5090)', () => {
    const comfortable = r.picks.find((p) => p.category === 'confortable');
    expect(comfortable?.stationId).toBe('mac');
  });

  it('≤ 3 picks, catégories distinctes, maxGpuCountExceeded = false', () => {
    expect(r.picks.length).toBeLessThanOrEqual(3);
    expect(new Set(r.picks.map((p) => p.category)).size).toBe(r.picks.length);
    expect(r.maxGpuCountExceeded).toBe(false);
  });
});
```

- [ ] **Step 2 : Exécuter — doit échouer**

```bash
npm test -- engine/index
```
Attendu : FAIL.

- [ ] **Step 3 : Implémentation**

`src/engine/index.ts` :
```ts
import type {
  HardwareCatalog, ModelSpec, QuantizationId, Scenario, SizingResult,
} from '../types';
import { estimateMemory } from './memory';
import { offeredTps } from './performance';
import { recommendPicks } from './recommend';

/**
 * API unique du moteur : dimensionnement complet à partir d'un modèle résolu,
 * d'un scénario de charge et du catalogue matériel. Pur, sans réseau ni React.
 */
export function computeSizing(
  spec: ModelSpec,
  scenario: Scenario,
  q: QuantizationId,
  catalog: HardwareCatalog,
): SizingResult {
  const { picks, maxGpuCountExceeded } = recommendPicks(spec, scenario, q, catalog);
  return {
    memory: estimateMemory(spec, scenario, q),
    offeredTps: offeredTps(scenario),
    picks,
    maxGpuCountExceeded,
  };
}

export { resolveModel, searchModels, buildSpec, ModelResolutionError } from './model';
export { estimateMemory, systemRamGB, weightsBytes, MAX_GPUS } from './memory';
export { BW_EFFICIENCY, PREFILL_ETA } from './performance';
export { recommendPicks } from './recommend';
```

- [ ] **Step 4 : Exécuter — tout le moteur doit être vert**

```bash
npm test
```
Attendu : 100 % PASS (types, model, memory, performance, recommend, index, hardware, App).
Vérification manuelle attendue pour le golden : viable = [2×5090/desk 8 529 €, Mac 9 999 €, 1×PRO/desk 13 146 €] → moins chère = 2×5090, équilibrée = PRO (proche médiane), confortable = Mac.

- [ ] **Step 5 : Commit**

```bash
git add src/engine/index.ts src/engine/index.test.ts
git commit -m "feat(engine): computeSizing orchestrateur + golden Qwen3-30B-A3B bout-en-bout"
```

---

### Task 9 : Composant ModelSearch (recherche HF avec autocomplétion)

**Files:**
- Create: `src/components/ModelSearch.tsx`
- Test: `src/components/ModelSearch.test.tsx`

**Interfaces:**
- Consumes: `searchModels(query: string): Promise<{id: string}[]>`, `resolveModel(id: string): Promise<ModelSpec>`, `ModelResolutionError` (exportés par `src/engine/index.ts`, Task 8) ; type `ModelSpec` (Task 2).
- Produces: `export default function ModelSearch(props: { onResolved: (spec: ModelSpec) => void; onError: (message: string) => void }): JSX.Element` — App.tsx (Task 10) l'utilise tel quel.

- [ ] **Step 1 : Écrire le test qui échoue (smoke, sans réseau)**

```tsx
// src/components/ModelSearch.test.tsx
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import ModelSearch from './ModelSearch';
import { resolveModel } from '../engine';
import { ModelResolutionError } from '../engine/model';

vi.mock('../engine', async (importOriginal) => ({
  ...(await importOriginal<typeof import('../engine')>()),
  resolveModel: vi.fn().mockRejectedValue(
    new ModelResolutionError('Modèle introuvable : x/y')
  ),
}));

describe('ModelSearch', () => {
  it('affiche le champ de recherche de modèle', () => {
    render(<ModelSearch onResolved={() => {}} onError={() => {}} />);
    expect(screen.getByLabelText(/modèle hugging face/i)).toBeInTheDocument();
    expect(
      screen.getByPlaceholderText(/ex\. Qwen\/Qwen3-30B-A3B/i)
    ).toBeInTheDocument();
  });

  it('affiche un bouton Réessayer quand la résolution échoue, et relance au clic', async () => {
    const onError = vi.fn();
    render(<ModelSearch onResolved={() => {}} onError={onError} />);
    const input = screen.getByLabelText(/modèle hugging face/i);
    fireEvent.change(input, { target: { value: 'x/y' } });
    fireEvent.blur(input);
    const retry = await screen.findByRole('button', { name: /réessayer/i });
    expect(onError).toHaveBeenCalledWith('Modèle introuvable : x/y');
    fireEvent.click(retry);
    await screen.findByRole('button', { name: /réessayer/i });
    expect(vi.mocked(resolveModel).mock.calls.length).toBe(2);
  });
});
```

- [ ] **Step 2 : Run test to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module './ModelSearch'` (ou résolution d'import échouée).

- [ ] **Step 3 : Écrire l'implémentation minimale**

```tsx
// src/components/ModelSearch.tsx
import { useEffect, useState } from 'react';
import { searchModels, resolveModel, ModelResolutionError } from '../engine';
import type { ModelSpec } from '../types';

interface Props {
  onResolved: (spec: ModelSpec) => void;
  onError: (message: string) => void;
}

export default function ModelSearch({ onResolved, onError }: Props) {
  const [query, setQuery] = useState('');
  const [hits, setHits] = useState<{ id: string }[]>([]);
  const [loading, setLoading] = useState(false);
  const [resolvedName, setResolvedName] = useState<string | null>(null);
  const [lastAttempt, setLastAttempt] = useState<string | null>(null);

  useEffect(() => {
    const trimmed = query.trim();
    if (trimmed.length < 3) {
      setHits([]);
      return;
    }
    const timer = setTimeout(() => {
      searchModels(trimmed)
        .then(setHits)
        .catch(() => setHits([]));
    }, 300);
    return () => clearTimeout(timer);
  }, [query]);

  async function pick(id: string): Promise<void> {
    setLoading(true);
    try {
      const spec = await resolveModel(id);
      setResolvedName(spec.name);
      setHits([]);
      setQuery(id);
      setLastAttempt(null);
      onResolved(spec);
    } catch (err) {
      setLastAttempt(id);
      onError(
        err instanceof ModelResolutionError
          ? err.message
          : 'Hugging Face injoignable — vérifie ta connexion, puis réessaie'
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section>
      <h2>Modèle</h2>
      <label htmlFor="model-search">Modèle Hugging Face</label>
      <input
        id="model-search"
        list="model-hits"
        placeholder="ex. Qwen/Qwen3-30B-A3B"
        value={query}
        onChange={(e) => {
          setResolvedName(null);
          setQuery(e.target.value);
        }}
        onBlur={() => {
          const id = query.trim();
          if (id.includes('/') && !resolvedName) void pick(id);
        }}
      />
      <datalist id="model-hits">
        {hits.map((h) => (
          <option key={h.id} value={h.id} />
        ))}
      </datalist>
      {loading && <p role="status">Résolution du modèle…</p>}
      {resolvedName && <p role="status">✅ {resolvedName}</p>}
      {lastAttempt && (
        <button type="button" onClick={() => void pick(lastAttempt)}>
          Réessayer
        </button>
      )}
    </section>
  );
}
```

- [ ] **Step 4 : Run test to verify it passes**

Run: `npm test`
Expected: PASS — tous les tests (Tasks 1-9) verts. Le smoke test ne déclenche aucun appel réseau (debounce 300 ms non écoulé pendant le rendu).

- [ ] **Step 5 : Commit**

```bash
git add src/components/ModelSearch.tsx src/components/ModelSearch.test.tsx
git commit -m "feat(ui): recherche de modèle HF avec autocomplétion et gestion d'erreurs"
```

---

### Task 10 : Formulaire de scénario + layout deux colonnes (App SANS ResultsPanel)

**Files:**
- Create: `src/components/ScenarioForm.tsx`
- Test: `src/components/ScenarioForm.test.tsx`
- Modify: `src/App.tsx` (remplace le placeholder Task 1 par la version 2 colonnes, SANS ResultsPanel)
- Modify: `src/App.css` (remplace le contenu vide par les styles complets)
- Modify: `src/App.test.tsx` (remplace le smoke test)

**Interfaces:**
- Consumes: types `Scenario`, `QuantizationId`, `ModelSpec`, `HardwareCatalog` + constantes `QUANTIZATIONS` (Task 2) ; `computeSizing(spec, scenario, quant, catalog): SizingResult` et `weightsBytes(paramsTotal, quant): number` (Task 8) ; composant `ModelSearch` (Task 9) ; `catalogJson` importé de `src/data/hardware.json` (Task 6).
- Produces: `export default function ScenarioForm(props: { scenario: Scenario; quant: QuantizationId; weightsGo: number | null; onScenarioChange: (patch: Partial<Scenario>) => void; onQuantChange: (q: QuantizationId) => void })` (câblé par App, Task 11 inchangé) ; App.tsx expose le state `sizing: SizingResult | null` et `scenario` que la Task 11 câblera à `ResultsPanel`.

- [ ] **Step 1 : Écrire les tests qui échouent**

```tsx
// src/components/ScenarioForm.test.tsx
import { describe, expect, it, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ScenarioForm from './ScenarioForm';
import type { Scenario } from '../types';
import { DEFAULT_SCENARIO } from './ScenarioForm';

describe('ScenarioForm', () => {
  const base: Scenario = { ...DEFAULT_SCENARIO };

  it('affiche le poids estimé en gigaoctets (format fr)', () => {
    render(
      <ScenarioForm
        scenario={base}
        quant="Q4_K_M"
        weightsGo={18.3}
        onScenarioChange={() => {}}
        onQuantChange={() => {}}
      />
    );
    expect(screen.getByText(/≈ 18,3 Go de poids/i)).toBeInTheDocument();
  });

  it('affiche le champ Utilisateurs simultanés', () => {
    render(
      <ScenarioForm
        scenario={base}
        quant="Q4_K_M"
        weightsGo={null}
        onScenarioChange={() => {}}
        onQuantChange={() => {}}
      />
    );
    expect(screen.getByLabelText(/utilisateurs simultanés/i)).toBeInTheDocument();
  });

  it('émet un patch quand un champ change', () => {
    const onScenarioChange = vi.fn();
    render(
      <ScenarioForm
        scenario={base}
        quant="Q4_K_M"
        weightsGo={null}
        onScenarioChange={onScenarioChange}
        onQuantChange={() => {}}
      />
    );
    fireEvent.change(screen.getByLabelText(/utilisateurs simultanés/i), {
      target: { value: '12' },
    });
    expect(onScenarioChange).toHaveBeenCalledWith({ concurrentUsers: 12 });
  });

  it('avertit quand le contexte est inférieur à entrée + sortie', () => {
    const tight: Scenario = {
      ...base,
      avgInputTokens: 9000,
      avgOutputTokens: 9000,
      maxContextTokens: 8192,
    };
    render(
      <ScenarioForm
        scenario={tight}
        quant="Q4_K_M"
        weightsGo={null}
        onScenarioChange={() => {}}
        onQuantChange={() => {}}
      />
    );
    expect(
      screen.getByText(/contexte maximum est inférieur/i)
    ).toBeInTheDocument();
  });
});
```

```tsx
// src/App.test.tsx (REMPLACE le smoke test Task 1)
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import App from './App';

describe('App', () => {
  it('affiche le titre et le formulaire de scénario', () => {
    render(<App />);
    expect(
      screen.getByText(/prédicteur llm — dimensionnement matériel/i)
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText(/utilisateurs simultanés/i)
    ).toBeInTheDocument();
  });

  it('affiche le bandeau d'hypothèses', () => {
    render(<App />);
    expect(screen.getByText(/estimations ±25-30/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2 : Run tests to verify they fail**

Run: `npm test`
Expected: FAIL — `Cannot find module './ScenarioForm'` ; App.test échoue (pas de label « Utilisateurs simultanés » ni bandeau).

- [ ] **Step 3 : Écrire ScenarioForm.tsx**

```tsx
// src/components/ScenarioForm.tsx
import type { QuantizationId, Scenario } from '../types';
import { QUANTIZATIONS } from '../types';

export const DEFAULT_SCENARIO: Scenario = {
  concurrentUsers: 10,
  totalUsers: 30,
  requestsPerUserPerMin: 3,
  avgInputTokens: 2000,
  avgOutputTokens: 500,
  maxContextTokens: 16384,
  ttftTargetSec: 2,
  minTpsPerUser: 20,
  safetyMarginPct: 20,
};

const CTX_OPTIONS = [4096, 8192, 16384, 32768, 65536, 131072];

interface Props {
  scenario: Scenario;
  quant: QuantizationId;
  weightsGo: number | null;
  onScenarioChange: (patch: Partial<Scenario>) => void;
  onQuantChange: (q: QuantizationId) => void;
}

export default function ScenarioForm({
  scenario,
  quant,
  weightsGo,
  onScenarioChange,
  onQuantChange,
}: Props) {
  const ctxTooSmall =
    scenario.maxContextTokens <
    scenario.avgInputTokens + scenario.avgOutputTokens;

  return (
    <section>
      <h2>Quantification</h2>
      <label htmlFor="quant">Quantification</label>
      <select
        id="quant"
        value={quant}
        onChange={(e) => onQuantChange(e.target.value as QuantizationId)}
      >
        {QUANTIZATIONS.map((q) => (
          <option key={q.id} value={q.id}>
            {q.label}
          </option>
        ))}
      </select>
      {weightsGo !== null && (
        <p>
          ≈{' '}
          {weightsGo.toLocaleString('fr-FR', { maximumFractionDigits: 1 })} Go de
          poids
        </p>
      )}

      <h2>Charge</h2>
      <label htmlFor="concurrent-users">Utilisateurs simultanés</label>
      <input
        id="concurrent-users"
        type="number"
        aria-label="Utilisateurs simultanés"
        min={1}
        value={scenario.concurrentUsers}
        onChange={(e) =>
          onScenarioChange({
            concurrentUsers: Math.max(1, Number(e.target.value) || 1),
          })
        }
      />

      <label htmlFor="total-users">Utilisateurs totaux</label>
      <input
        id="total-users"
        type="number"
        aria-label="Utilisateurs totaux"
        min={1}
        value={scenario.totalUsers}
        onChange={(e) =>
          onScenarioChange({
            totalUsers: Math.max(1, Number(e.target.value) || 1),
          })
        }
      />

      <label htmlFor="req-min">Requêtes / utilisateur / minute</label>
      <input
        id="req-min"
        type="number"
        aria-label="Requêtes / utilisateur / minute"
        min={0}
        value={scenario.requestsPerUserPerMin}
        onChange={(e) =>
          onScenarioChange({
            requestsPerUserPerMin: Math.max(0, Number(e.target.value) || 0),
          })
        }
      />

      <label htmlFor="in-tokens">Tokens entrée moyens</label>
      <input
        id="in-tokens"
        type="number"
        aria-label="Tokens entrée moyens"
        min={1}
        value={scenario.avgInputTokens}
        onChange={(e) =>
          onScenarioChange({
            avgInputTokens: Math.max(1, Number(e.target.value) || 1),
          })
        }
      />

      <label htmlFor="out-tokens">Tokens sortie moyens</label>
      <input
        id="out-tokens"
        type="number"
        aria-label="Tokens sortie moyens"
        min={1}
        value={scenario.avgOutputTokens}
        onChange={(e) =>
          onScenarioChange({
            avgOutputTokens: Math.max(1, Number(e.target.value) || 1),
          })
        }
      />

      <label htmlFor="ctx">Contexte maximum</label>
      <select
        id="ctx"
        aria-label="Contexte maximum"
        value={scenario.maxContextTokens}
        onChange={(e) =>
          onScenarioChange({ maxContextTokens: Number(e.target.value) })
        }
      >
        {CTX_OPTIONS.map((c) => (
          <option key={c} value={c}>
            {c.toLocaleString('fr-FR')} tokens
          </option>
        ))}
      </select>
      {ctxTooSmall && (
        <p role="alert">
          ⚠️ Le contexte maximum est inférieur aux tokens d'entrée + sortie de
          la requête moyenne.
        </p>
      )}

      <h2>Cibles &amp; marge</h2>
      <label htmlFor="ttft">TTFT max (s)</label>
      <input
        id="ttft"
        type="number"
        aria-label="TTFT max (s)"
        min={0.1}
        step={0.1}
        value={scenario.ttftTargetSec}
        onChange={(e) =>
          onScenarioChange({
            ttftTargetSec: Math.max(0.1, Number(e.target.value) || 0.1),
          })
        }
      />

      <label htmlFor="tps">Débit min / utilisateur (tok/s)</label>
      <input
        id="tps"
        type="number"
        aria-label="Débit min / utilisateur (tok/s)"
        min={1}
        value={scenario.minTpsPerUser}
        onChange={(e) =>
          onScenarioChange({
            minTpsPerUser: Math.max(1, Number(e.target.value) || 1),
          })
        }
      />

      <label htmlFor="margin">Marge de sécurité (%)</label>
      <input
        id="margin"
        type="number"
        aria-label="Marge de sécurité (%)"
        min={0}
        max={100}
        step={5}
        value={scenario.safetyMarginPct}
        onChange={(e) =>
          onScenarioChange({
            safetyMarginPct: Math.min(
              100,
              Math.max(0, Number(e.target.value) || 0)
            ),
          })
        }
      />
    </section>
  );
}
```

- [ ] **Step 4 : Remplacer App.tsx par la version deux colonnes (SANS ResultsPanel)**

```tsx
// src/App.tsx
import { useMemo, useState } from 'react';
import type {
  HardwareCatalog,
  ModelSpec,
  QuantizationId,
  Scenario,
} from './types';
import { computeSizing, weightsBytes } from './engine';
import catalogJson from './data/hardware.json';
import ModelSearch from './components/ModelSearch';
import ScenarioForm, { DEFAULT_SCENARIO } from './components/ScenarioForm';

const catalog = catalogJson as unknown as HardwareCatalog;

export default function App() {
  const [spec, setSpec] = useState<ModelSpec | null>(null);
  const [scenario, setScenario] = useState<Scenario>(DEFAULT_SCENARIO);
  const [quant, setQuant] = useState<QuantizationId>('Q4_K_M');
  const [error, setError] = useState<string | null>(null);

  const sizing = useMemo(
    () => (spec ? computeSizing(spec, scenario, quant, catalog) : null),
    [spec, scenario, quant]
  );

  const weightsGo = spec
    ? weightsBytes(spec.paramsTotal, quant) / 1e9
    : null;

  return (
    <main>
      <h1>Prédicteur LLM — dimensionnement matériel</h1>
      <p className="banner">
        Estimations ±25-30 % — hypothèses : KV en FP16, continuous batching,
        interconnexion multi-GPU non modélisée au-delà du partage poids/KV —
        prix relevés du {catalog.priceDate}.
      </p>
      <div className="layout">
        <div className="col-form">
          <ModelSearch onResolved={setSpec} onError={setError} />
          {spec?.isMoe && (
            <p className="badge">
              MoE détecté : {spec.paramsTotal / 1e9} Md totaux, ~
              {spec.paramsActive / 1e9} Md actifs par token
            </p>
          )}
          {(spec?.derived.kvHeads ||
            spec?.derived.headDim ||
            spec?.derived.moe) && (
            <p className="badge">⚠️ Valeurs déduites (config incomplète)</p>
          )}
          <ScenarioForm
            scenario={scenario}
            quant={quant}
            weightsGo={weightsGo}
            onScenarioChange={(patch) =>
              setScenario((s) => ({ ...s, ...patch }))
            }
            onQuantChange={setQuant}
          />
          {error && (
            <p role="alert" className="error">
              {error}
            </p>
          )}
        </div>
        <div className="col-results">
          <p>Résous un modèle pour voir le dimensionnement.</p>
        </div>
      </div>
    </main>
  );
}
```

- [ ] **Step 5 : Remplacer App.css par les styles complets**

```css
/* src/App.css */
.banner {
  background: #fef3c7;
  border: 1px solid #f59e0b;
  border-radius: 8px;
  padding: 0.5rem 0.9rem;
  font-size: 0.9rem;
}

.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 2rem;
  margin-top: 1.5rem;
}

@media (max-width: 800px) {
  .layout {
    grid-template-columns: 1fr;
  }
}

section {
  margin-bottom: 1.5rem;
}

label {
  display: block;
  margin-top: 0.7rem;
  font-weight: 600;
  font-size: 0.9rem;
}

input,
select {
  margin-top: 0.25rem;
  width: 100%;
  max-width: 340px;
  padding: 0.4rem;
  font-size: 1rem;
}

.badge {
  font-size: 0.85rem;
  background: #e0e7ff;
  border-radius: 6px;
  padding: 0.3rem 0.6rem;
  display: inline-block;
}

.error,
[role='alert'] {
  color: #b91c1c;
  font-weight: 600;
}
```

- [ ] **Step 6 : Run tests to verify they pass**

Run: `npm test`
Expected: PASS — tous les tests (Tasks 1-10) verts, y compris les 4 nouveaux ScenarioForm et les 2 App.

- [ ] **Step 7 : Commit**

```bash
git add src/components/ScenarioForm.tsx src/components/ScenarioForm.test.tsx src/App.tsx src/App.css src/App.test.tsx
git commit -m "feat(ui): formulaire de scénario + layout deux colonnes"
```

---

### Task 11 : Panneau de résultats (camembert, performances, recommandations, BOM) + câblage App

**Files:**
- Create: `src/components/DonutChart.tsx`
- Create: `src/components/ResultsPanel.tsx`
- Test: `src/components/ResultsPanel.test.tsx`
- Modify: `src/App.tsx` (import + câblage ResultsPanel dans col-results)

**Interfaces:**
- Consumes: types `Scenario`, `SizingResult` (Task 2) ; `computeSizing`, `estimateMemory` (Task 8, pour le test) ; state `sizing`/`scenario` d'App (Task 10).
- Produces: `export interface DonutSegment { label: string; value: number; color: string }` ; `export default function DonutChart(props: { segments: DonutSegment[]; centerLabel: string })` ; `export default function ResultsPanel(props: { sizing: SizingResult; scenario: Scenario })` (c'est la version finale d'App).

- [ ] **Step 1 : Écrire le test qui échoue**

```tsx
// src/components/ResultsPanel.test.tsx
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import ResultsPanel from './ResultsPanel';
import { computeSizing, estimateMemory } from '../engine';
import type { HardwareCatalog, ModelSpec, Scenario } from '../types';

const QWEN30B: ModelSpec = {
  id: 'Qwen/Qwen3-30B-A3B',
  name: 'Qwen3-30B-A3B',
  paramsTotal: 30.5e9,
  paramsActive: 3.3e9,
  isMoe: true,
  layers: 48,
  kvHeads: 4,
  headDim: 128,
  hiddenSize: 2048,
  derived: { kvHeads: false, headDim: false, moe: false },
};

const SC: Scenario = {
  concurrentUsers: 10,
  totalUsers: 30,
  requestsPerUserPerMin: 3,
  avgInputTokens: 2000,
  avgOutputTokens: 500,
  maxContextTokens: 16384,
  ttftTargetSec: 2,
  minTpsPerUser: 20,
  safetyMarginPct: 20,
};

const MINI_CATALOG: HardwareCatalog = {
  gpus: [
    { id: 'rtx5090', name: 'RTX 5090', vramGB: 32, usableVramGB: 29.8, bandwidthGBps: 1792, flopsFP16: 419, tdpW: 575, priceEur: 2799 },
    { id: 'rtxpro6000', name: 'RTX PRO 6000', vramGB: 96, usableVramGB: 92, bandwidthGBps: 1793, flopsFP16: 470, tdpW: 600, priceEur: 9999 },
  ],
  stations: [
    { id: 'mac', name: 'Mac Studio M3 Ultra 256 Go', memoryGB: 256, bandwidthGBps: 819, flopsFP16: 80, priceEur: 9999 },
  ],
  platforms: [
    { id: 'desk', name: 'Desktop 2 GPU', gpuSlots: 2, priceEur: 2499 },
    { id: 'rack', name: 'Rack 8 GPU', gpuSlots: 8, priceEur: 24999 },
  ],
  ramPricePerGB: 4.5,
  priceDate: '2026-09',
};

describe('ResultsPanel', () => {
  it('affiche la charge offerte et la recommandation la moins chère', () => {
    const sizing = computeSizing(QWEN30B, SC, 'Q4_K_M', MINI_CATALOG);
    render(<ResultsPanel sizing={sizing} scenario={SC} />);
    expect(screen.getAllByText(/charge offerte/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/la moins chère/i)).toBeInTheDocument();
  });

  it("affiche le message « plus de 8 GPU » quand aucune config ne tient", () => {
    const memory = estimateMemory(QWEN30B, SC, 'Q4_K_M');
    render(
      <ResultsPanel
        sizing={{ memory, offeredTps: 100, picks: [], maxGpuCountExceeded: true }}
        scenario={SC}
      />
    );
    expect(screen.getByText(/plus de 8 GPU/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2 : Run test to verify it fails**

Run: `npm test`
Expected: FAIL — `Cannot find module './ResultsPanel'`.

- [ ] **Step 3 : Écrire DonutChart.tsx**

```tsx
// src/components/DonutChart.tsx
export interface DonutSegment {
  label: string;
  value: number;
  color: string;
}

interface Props {
  segments: DonutSegment[];
  centerLabel: string;
}

export default function DonutChart({ segments, centerLabel }: Props) {
  const total = segments.reduce((sum, s) => sum + s.value, 0) || 1;
  const R = 42;
  const C = 2 * Math.PI * R;
  let offset = 0;

  return (
    <svg
      viewBox="0 0 100 100"
      width={180}
      height={180}
      role="img"
      aria-label={centerLabel}
    >
      {segments.map((s) => {
        const frac = s.value / total;
        const dashOffset = -offset * C;
        offset += frac;
        return (
          <circle
            key={s.label}
            cx={50}
            cy={50}
            r={R}
            fill="none"
            stroke={s.color}
            strokeWidth={12}
            strokeDasharray={`${frac * C} ${C}`}
            strokeDashoffset={dashOffset}
            transform="rotate(-90 50 50)"
          />
        );
      })}
      <text x={50} y={54} textAnchor="middle" fontSize={10}>
        {centerLabel}
      </text>
    </svg>
  );
}
```

- [ ] **Step 4 : Écrire ResultsPanel.tsx**

```tsx
// src/components/ResultsPanel.tsx
import type { Scenario, SizingResult } from '../types';
import DonutChart from './DonutChart';

interface Props {
  sizing: SizingResult;
  scenario: Scenario;
}

const eur = (n: number): string =>
  n.toLocaleString('fr-FR', {
    style: 'currency',
    currency: 'EUR',
    maximumFractionDigits: 0,
  });

const go = (bytes: number): string =>
  `${(bytes / 1e9).toLocaleString('fr-FR', { maximumFractionDigits: 1 })} Go`;

const mark = (ok: boolean): string => (ok ? '✅' : '❌');

export default function ResultsPanel({ sizing, scenario }: Props) {
  const { memory, offeredTps, picks, maxGpuCountExceeded } = sizing;

  return (
    <section aria-label="Résultats du dimensionnement">
      <h2>Mémoire requise</h2>
      <DonutChart
        centerLabel={go(memory.vramRequiredBytes)}
        segments={[
          { label: 'Poids', value: memory.weightsBytes, color: '#3b82f6' },
          { label: 'KV cache', value: memory.kvBytes, color: '#f59e0b' },
          { label: 'Overhead', value: memory.overheadBytes, color: '#94a3b8' },
        ]}
      />
      <ul>
        <li>Poids : {go(memory.weightsBytes)}</li>
        <li>KV cache : {go(memory.kvBytes)}</li>
        <li>
          VRAM totale requise (marge incluse) : {go(memory.vramRequiredBytes)}
        </li>
      </ul>

      <h2>Charge</h2>
      <p>
        Charge offerte : {Math.round(offeredTps).toLocaleString('fr-FR')} tok/s
      </p>

      {picks.length === 0 ? (
        <p role="alert">
          {maxGpuCountExceeded
            ? 'Aucune configuration du catalogue ne tient ce scénario (il faudrait plus de 8 GPU). Pistes : réduis le contexte ou les simultanés, ou quantifie davantage.'
            : 'Aucune configuration viable ne respecte tes cibles. Pistes : quantifie davantage, réduis le contexte ou les simultanés.'}
        </p>
      ) : (
        picks.map((p) => (
          <article className="pick" key={`${p.label}-${p.category}`}>
            <h3>
              {p.category[0].toUpperCase() + p.category.slice(1)} — {p.label}
            </h3>
            <ul>
              <li>
                TTFT estimé {p.perf.ttftSec.toFixed(2)} s {mark(p.perf.meetsTtft)}{' '}
                (cible ≤ {scenario.ttftTargetSec} s)
              </li>
              <li>
                Débit / utilisateur{' '}
                {Math.round(p.perf.tpsPerUser).toLocaleString('fr-FR')} tok/s{' '}
                {mark(p.perf.meetsTps)} (cible ≥ {scenario.minTpsPerUser})
              </li>
              <li>
                Capacité agrégée{' '}
                {Math.round(p.perf.aggregateTps).toLocaleString('fr-FR')} tok/s —
                charge offerte {Math.round(offeredTps).toLocaleString('fr-FR')}{' '}
                tok/s {mark(p.perf.absorbsLoad)}
              </li>
              <li>Marge VRAM restante {p.vramHeadroomPct.toFixed(0)} %</li>
              {p.ramGB > 0 && <li>RAM système : {p.ramGB} Go</li>}
            </ul>
            <table>
              <caption>Nomenclature — total {eur(p.totalPriceEur)}</caption>
              <thead>
                <tr>
                  <th>Élément</th>
                  <th>Prix unitaire</th>
                  <th>Qté</th>
                  <th>Total</th>
                </tr>
              </thead>
              <tbody>
                {p.bom.map((l) => (
                  <tr key={l.label}>
                    <td>{l.label}</td>
                    <td>{eur(l.unitPriceEur)}</td>
                    <td>{l.qty}</td>
                    <td>{eur(l.lineTotalEur)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </article>
        ))
      )}
    </section>
  );
}
```

- [ ] **Step 5 : Câbler ResultsPanel dans App.tsx**

Deux changements exacts dans `src/App.tsx` (le reste est inchangé) :

1. Ajouter l'import après ceux de ScenarioForm :

```tsx
import ResultsPanel from './components/ResultsPanel';
```

2. Remplacer le contenu de la colonne résultats :

```tsx
        <div className="col-results">
          {sizing ? (
            <ResultsPanel sizing={sizing} scenario={scenario} />
          ) : (
            <p>Résous un modèle pour voir le dimensionnement.</p>
          )}
        </div>
```

- [ ] **Step 6 : Run tests to verify they pass**

Run: `npm test`
Expected: PASS — tous les tests (Tasks 1-11) verts, y compris les 2 nouveaux ResultsPanel.

- [ ] **Step 7 : Commit**

```bash
git add src/components/DonutChart.tsx src/components/ResultsPanel.tsx src/components/ResultsPanel.test.tsx src/App.tsx
git commit -m "feat(ui): panneau résultats — camembert, performances, recommandations, BOM"
```

---

### Task 12 : Calibration des constantes de performance sur benchmarks publics

**Files:**
- Modify: `src/engine/performance.ts` (constantes + commentaires sources)
- Modify: `src/engine/performance.test.ts` (valeurs golden recalculées si constantes changées)
- Modify: `src/engine/index.test.ts` (valeurs golden recalculées si constantes changées)

**Interfaces:**
- Consumes: formules roofline des Tasks 5 et 8 (inchangées).
- Produces: valeurs finales calibrées de `BW_EFFICIENCY` et `PREFILL_ETA`, avec sources citées en commentaire. Aucune signature ne change.

Ce n'est pas une tâche TDD classique : on ajuste des constantes de calibration sur des données publiées, puis on met à jour les valeurs attendues des tests golden existants. Les tests doivent rester verts à la fin — on ne modifie jamais les formules pour faire passer un test.

- [ ] **Step 1 : Rechercher des benchmarks publics**

Recherches web à effectuer (jina.ai, firecrawl, ou websearch) :
- « llama.cpp benchmark RTX 4090 8B Q4_K_M tokens per second »
- « RTX 5090 llama.cpp benchmark tokens per second »
- « DGX Spark llama.cpp tokens per second benchmark »

- [ ] **Step 2 : Comparer au modèle roofline**

Point de référence clé : un modèle 8B Q4_K_M en flux unique sur RTX 4090 tourne autour de 100-120 tok/s dans les benchmarks publiés (llama.cpp). Notre roofline avec B=1 et ctx_moyen≈2250 : octets/pas ≈ 8,03e9×0,6 + 131072×2250 ≈ 5,11e9 octets ; avec BW=1008 Go/s : BW_EFFICIENCY=0,85 ⇒ ≈168 tok/s (optimiste) ; ≈110 tok/s publiés ⇒ BW_EFFICIENCY ≈ 0,55-0,60. Rappel : le batching continu améliore l'utilisation de la bande passante, ce qui justifie une valeur au-dessus du single-stream.

- [ ] **Step 3 : Choisir les constantes**

Choisir `BW_EFFICIENCY ∈ [0,55, 0,85]` et `PREFILL_ETA ∈ [0,30, 0,50]` au mieux des points publiés trouvés au Step 1. Si aucun benchmark fiable n'est trouvé : conserver 0,85/0,40 et noter « non calibré — valeurs par défaut » dans le commentaire. Pas de placeholder dans le code.

- [ ] **Step 4 : Mettre à jour performance.ts + recalculer les valeurs golden**

Mettre à jour les constantes dans `src/engine/performance.ts` avec un commentaire citant les sources (URL + valeur publiée + date). Recalculer les valeurs attendues des tests golden Task 5 et Task 8 avec les formules :

- `tpsParUtilisateur = gpuCount × BW × BW_EFFICIENCY × 1e9 / (paramsActive × bpw/8 + concurrentUsers × kvParToken × ctx_moyen)`
- `ttft = 2 × paramsActive × avgInputTokens / (flopsFP16 × 1e12 × PREFILL_ETA)`

puis mettre à jour les `toBeCloseTo` correspondants dans `performance.test.ts` et `index.test.ts`.

- [ ] **Step 5 : Run tests to verify they pass**

Run: `npm test`
Expected: PASS — tous les tests verts avec les constantes calibrées.

- [ ] **Step 6 : Commit**

```bash
git add src/engine/performance.ts src/engine/performance.test.ts src/engine/index.test.ts
git commit -m "chore(engine): calibration BW_EFFICIENCY/PREFILL_ETA sur benchmarks publics (refs en commentaire)"
```

---

### Task 13 : CI GitHub Pages + README + vérification finale

**Files:**
- Create: `.github/workflows/deploy.yml`
- Create: `README.md`

**Interfaces:**
- Consumes: rien de nouveau (build/test existants des Tasks 1-12).
- Produces: pipeline de déploiement GitHub Pages (push sur main → build + test + publish) et documentation du repo. C'est la dernière tâche du plan.

- [ ] **Step 1 : Créer le workflow de déploiement**

```yaml
# .github/workflows/deploy.yml
name: Deploy GitHub Pages

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
      - run: npm ci
      - run: npm test
      - run: npm run build
      - uses: actions/upload-pages-artifact@v3
        with:
          path: dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2 : Créer le README**

```markdown
# Prédicteur de dimensionnement matériel LLM

Estime le matériel nécessaire (GPU, VRAM, RAM, plateforme) pour servir un modèle Hugging Face quantifié à un nombre d'utilisateurs en parallèle, et propose des configurations achetables avec leurs prix en euros.

> ⚠️ **Estimations ±25-30 %** — c'est un outil d'aide à la décision, pas un benchmark.

## Développement

```bash
npm ci        # installer les dépendances
npm run dev   # serveur de développement Vite
npm test      # tests unitaires (vitest)
npm run build # build de production (tsc + vite build → dist/)
```

## Mettre les prix à jour

Édite `src/data/hardware.json` : les prix sont en euros, indicatifs, et datés via le champ `priceDate`. Mets à jour les valeurs, puis commit — rien d'autre à faire.

## Déploiement

Push sur `main` : l'action GitHub build le site, exécute les tests et publie sur GitHub Pages. Dans les paramètres du repo : **Settings → Pages → Source = GitHub Actions**.

## Méthode

Les formules de dimensionnement (poids, KV cache, roofline de décode, TTFT) sont documentées dans [`docs/superpowers/specs/2026-09-04-llm-hardware-predictor-design.md`](docs/superpowers/specs/2026-09-04-llm-hardware-predictor-design.md).
```

- [ ] **Step 3 : Vérification finale du plan complet**

Run: `npm ci && npm test && npm run build`
Expected: exit 0 — toutes les dépendances installées, tous les tests au vert, build produit dans `dist/`. Puis `git status` → arbre propre (tout est committé).

- [ ] **Step 4 : Commit**

```bash
git add .github/workflows/deploy.yml README.md
git commit -m "ci: déploiement GitHub Pages + README"
```

---

## Checklist de fin de plan

Une fois les 13 tâches exécutées :

- [ ] `npm ci && npm test && npm run build` → exit 0
- [ ] `git log --oneline` → un commit par tâche, messages en français conformes
- [ ] `git status` → arbre propre
- [ ] Le site est prêt pour GitHub Pages : créer le repo GitHub (public ou private), `git remote add origin …`, `git push -u origin main`, puis dans Settings → Pages sélectionner Source = GitHub Actions.

