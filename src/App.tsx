import { useCallback, useEffect, useState } from 'react';
import {
  HfError,
  HfRateLimitError,
  type ModelSearchHit,
  type ResolvedModel,
  resolveModel,
  searchModels,
} from './api/hf';
import { AssumptionsBanner } from './components/AssumptionsBanner';
import { LoadForm, type FormState } from './components/LoadForm';
import { MemoryCard } from './components/MemoryCard';
import { ModelSearch } from './components/ModelSearch';
import { PerformanceCard } from './components/PerformanceCard';
import { RecommendationCard } from './components/RecommendationCard';
import {
  type RecommendationResult,
  recommendHardware,
} from './engine/recommendations';
import type { Scenario } from './engine/types';
import { catalog } from './lib/catalog';

const DEFAULT_FORM: FormState = {
  quant: 'Q4_K_M',
  simultaneous: '10',
  totalUsers: '30',
  reqPerUserPerMin: '3',
  inputTokens: '2000',
  outputTokens: '500',
  contextMax: '16384',
  ttftTargetSec: '2',
  minTpsPerUser: '20',
  safetyMargin: '0.2',
};

function parsePositive(value: string, fallback: number): number {
  const n = Number(value);
  return Number.isFinite(n) && n >= 0 ? n : fallback;
}

function parseOptionalPositive(value: string): number | undefined {
  const trimmed = value.trim();
  if (trimmed === '') return undefined;
  const n = Number(trimmed);
  return Number.isFinite(n) && n > 0 ? n : undefined;
}

function buildScenario(form: FormState): Scenario {
  return {
    quant: form.quant,
    simultaneous: Math.max(1, parsePositive(form.simultaneous, 1)),
    totalUsers: parsePositive(form.totalUsers, 1),
    reqPerUserPerMin: parsePositive(form.reqPerUserPerMin, 1),
    inputTokens: parsePositive(form.inputTokens, 512),
    outputTokens: parsePositive(form.outputTokens, 256),
    contextMax: parsePositive(form.contextMax, 4096),
    safetyMargin: parsePositive(form.safetyMargin, 0.2),
    ttftTargetSec: parseOptionalPositive(form.ttftTargetSec),
    minTpsPerUser: parseOptionalPositive(form.minTpsPerUser),
  };
}

type ModelStatus =
  | { kind: 'idle' }
  | { kind: 'searching' }
  | { kind: 'resolving' }
  | { kind: 'ready'; model: ResolvedModel }
  | { kind: 'error'; error: HfError };

export default function App() {
  const [query, setQuery] = useState('');
  const [hits, setHits] = useState<readonly ModelSearchHit[]>([]);
  const [modelStatus, setModelStatus] = useState<ModelStatus>({ kind: 'idle' });
  const [cache, setCache] = useState<Map<string, ResolvedModel>>(new Map());
  const [retrying, setRetrying] = useState(false);
  const [form, setForm] = useState<FormState>(DEFAULT_FORM);
  const [hasCalculated, setHasCalculated] = useState(false);
  const [recommendationResult, setRecommendationResult] =
    useState<RecommendationResult | null>(null);
  const [computeError, setComputeError] = useState<Error | null>(null);

  const handleFormChange = useCallback((patch: Partial<FormState>) => {
    setForm((prev) => ({ ...prev, ...patch }));
  }, []);

  // Autocomplete search with 300 ms debounce and per-batch AbortController.
  useEffect(() => {
    const controller = new AbortController();
    const trimmed = query.trim();

    if (trimmed.length < 3) {
      setHits([]);
      setModelStatus((prev) =>
        prev.kind === 'searching' || prev.kind === 'error' ? { kind: 'idle' } : prev,
      );
      return;
    }

    const timer = setTimeout(async () => {
      setModelStatus((prev) => (prev.kind === 'idle' ? { kind: 'searching' } : prev));
      try {
        const results = await searchModels(trimmed, { signal: controller.signal });
        if (!controller.signal.aborted) {
          setHits(results);
        }
      } catch (err) {
        if (!controller.signal.aborted && err instanceof HfError) {
          setModelStatus({ kind: 'error', error: err });
        }
      } finally {
        if (!controller.signal.aborted) {
          setModelStatus((prev) => (prev.kind === 'searching' ? { kind: 'idle' } : prev));
        }
      }
    }, 300);

    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [query]);

  const resolveModelById = useCallback(
    async (id: string) => {
      const cached = cache.get(id);
      if (cached) {
        setModelStatus({ kind: 'ready', model: cached });
        setHasCalculated(false);
        return;
      }

      setModelStatus({ kind: 'resolving' });
      setRetrying(false);
      try {
        const resolved = await resolveModel(id);
        setCache((prev) => new Map(prev).set(id, resolved));
        setModelStatus({ kind: 'ready', model: resolved });
        setHasCalculated(false);
      } catch (err) {
        if (err instanceof HfError) {
          setModelStatus({ kind: 'error', error: err });
          if (err instanceof HfRateLimitError) {
            setRetrying(true);
            setTimeout(() => {
              setRetrying(false);
              void resolveModelById(id);
            }, 3000);
          }
        }
      }
    },
    [cache],
  );

  const handleSelectHit = useCallback(
    (id: string) => {
      void resolveModelById(id);
    },
    [resolveModelById],
  );

  const handleResolveDirect = useCallback(() => {
    const trimmed = query.trim();
    if (trimmed.length >= 3) {
      void resolveModelById(trimmed);
    }
  }, [query, resolveModelById]);

  const handleRetry = useCallback(() => {
    const trimmed = query.trim();
    if (modelStatus.kind === 'error' && trimmed.length >= 3) {
      void resolveModelById(trimmed);
    }
  }, [modelStatus, query, resolveModelById]);

  // Recompute recommendations whenever the model or form changes (after the
  // first voluntary calculation).
  useEffect(() => {
    if (modelStatus.kind !== 'ready' || !hasCalculated) {
      setRecommendationResult(null);
      setComputeError(null);
      return;
    }
    try {
      const scenario = buildScenario(form);
      const result = recommendHardware(modelStatus.model, scenario, catalog);
      setRecommendationResult(result);
      setComputeError(null);
    } catch (err) {
      setRecommendationResult(null);
      setComputeError(err instanceof Error ? err : new Error(String(err)));
    }
  }, [modelStatus, form, hasCalculated]);

  const selectedModel = modelStatus.kind === 'ready' ? modelStatus.model : null;
  const modelError = modelStatus.kind === 'error' ? modelStatus.error : null;

  const primaryRecommendation =
    recommendationResult?.status === 'viable'
      ? recommendationResult.recommendations[0]
      : undefined;

  return (
    <div className="app">
      <header className="page-header">
        <h1>Dimensionnement matériel LLM</h1>
        <p className="page-subtitle">
          Estime la VRAM, la RAM, les performances et le coût d’une infrastructure locale pour
          exécuter un modèle de langage.
        </p>
      </header>

      <main className="layout">
        <section className="form-column" aria-label="Paramètres">
          <ModelSearch
            query={query}
            onQueryChange={setQuery}
            hits={hits}
            isSearching={modelStatus.kind === 'searching'}
            selectedModel={selectedModel}
            error={modelError}
            retrying={retrying}
            onSelectHit={handleSelectHit}
            onResolveDirect={handleResolveDirect}
            onRetry={handleRetry}
          />
          <LoadForm form={form} onChange={handleFormChange} model={selectedModel} />
        </section>

        <section className="results-column" aria-label="Résultats">
          {modelStatus.kind === 'resolving' && (
            <div className="card empty-state">
              <p>Résolution du modèle sur Hugging Face…</p>
            </div>
          )}

          {modelStatus.kind === 'ready' && !hasCalculated && (
            <div className="card empty-state stack center">
              <p>
                Modèle <strong>{modelStatus.model.name}</strong> prêt.
              </p>
              <button
                type="button"
                className="button button-primary"
                onClick={() => setHasCalculated(true)}
              >
                Calculer les recommandations
              </button>
            </div>
          )}

          {modelStatus.kind !== 'resolving' && modelStatus.kind !== 'ready' && (
            <div className="card empty-state">
              <p>
                Choisis un modèle dans le formulaire de gauche pour obtenir un dimensionnement.
              </p>
            </div>
          )}

          {computeError && (
            <div className="card error-state" role="alert">
              <p>Erreur lors du calcul : {computeError.message}</p>
            </div>
          )}

          {recommendationResult?.status === 'viable' && primaryRecommendation && (
            <>
              <MemoryCard sizing={primaryRecommendation.pick.sizing} />
              <PerformanceCard sizing={primaryRecommendation.pick.sizing} />
              <div className="recommendations-stack">
                <h2 className="section-title">Recommandations</h2>
                <div className="stack">
                  {recommendationResult.recommendations.map((rec) => (
                    <RecommendationCard key={rec.role} recommendation={rec} />
                  ))}
                </div>
              </div>
            </>
          )}

          {recommendationResult?.status === 'aucune' && (
            <div className="card no-result" role="status">
              <h2 className="card-title">Aucune configuration viable</h2>
              <ul className="reasons-list">
                {recommendationResult.reasons.map((reason) => (
                  <li key={reason}>{reason}</li>
                ))}
              </ul>
            </div>
          )}

          <AssumptionsBanner version={catalog.version} />
        </section>
      </main>
    </div>
  );
}
