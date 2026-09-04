/**
 * Hugging Face resolution client (plan step 3).
 *
 * Two entry points for the UI:
 * - `searchModels`: autocomplete against `GET /api/models?search=…` filtered
 *   on `pipeline_tag=text-generation` (the ~300 ms debounce and the AbortSignal
 *   churn live in the UI, not here);
 * - `resolveModel`: fetches `GET /api/models/{id}` (for `safetensors.total`,
 *   all MoE experts included) and `{id}/resolve/main/config.json` (for the
 *   architecture fields), then maps them onto the engine's `ModelSpec`.
 *
 * Every documented fallback is flagged so the UI can show a
 * "valeurs déduites" badge (plan section "Résolution du modèle (HF)"):
 * - `kv_h`  ← `num_attention_heads` when `num_key_value_heads` is absent;
 * - `d_h`   ← `hidden_size / num_attention_heads` when `head_dim` is absent;
 * - a MoE whose expert structure is missing is flagged `moeTreatedAsDense`
 *   and stays dense for the engine (design §7: "traité comme dense +
 *   avertissement"), never a silent guess;
 * - for MoE, `expertSize` is resolved so that the engine formula
 *   (`num_experts_per_tok × expertSize`) reproduces the TRUE active
 *   parameters per token: active-expert FFN + attention + embeddings
 *   (+ lm_head when untied) + router. FFN-only undercounts by ~45 % for
 *   Qwen3-30B-A3B (calibration 2026-09-04, report §8). A missing
 *   `num_attention_heads` or `vocab_size` zeroes that term and flags
 *   `activeParamsPartial` — a visible lower bound, never a silent guess.
 *
 * Failure policy (design §7 - "pas de fallback caché"): every failure mode
 * throws a typed `HfError` subclass with a French message. HF being slow,
 * rate-limiting us or being down is NEVER converted into a result.
 *
 * No in-memory cache here on purpose: the design's "cache client en mémoire"
 * belongs to the UI wiring step, where freshness and invalidation are decided.
 */
import { activeParamsPerToken as computeActiveParams } from '../engine/formulas';
import type { ModelSpec } from '../engine/types';

/** Hugging Face public API root (no key required). */
export const HF_BASE_URL = 'https://huggingface.co';

/** Deadline for every HF request, enforced through AbortSignal. */
export const DEFAULT_TIMEOUT_MS = 10_000;

/** Minimum search query length before hitting the network. */
const SEARCH_MIN_QUERY_LENGTH = 3;

/** Autocomplete result cap (plan: "limit=10"). */
const SEARCH_LIMIT = 10;

/** Projection matrices stored per expert MLP: gate, up and down. */
const EXPERT_MATRICES = 3;

/** Options shared by every HF client call. */
export interface HfRequestOptions {
  /** Caller-owned cancellation signal (UI debounce, unmount, retry...). */
  readonly signal?: AbortSignal;
  /** Deadline override, for tests only. */
  readonly timeoutMs?: number;
}

/** One autocomplete entry: repo id (`org/model`) and short display name. */
export interface ModelSearchHit {
  readonly id: string;
  /** Short model name: the segment after the org slash of the repo id. */
  readonly name: string;
}

/**
 * Engine `ModelSpec` enriched with everything the UI needs that the engine
 * does not consume: identity, the active-parameter figure (A) pre-computed
 * through the engine formula, and the "valeurs déduites" / MoE-dense flags.
 */
export interface ResolvedModel extends ModelSpec {
  readonly id: string;
  readonly name: string;
  /** A - active params/token, engine formula (MoE: perTok x expertSize, else N). */
  readonly activeParamsPerToken: number;
  /** True when kvHeads came from the `num_attention_heads` fallback. */
  readonly kvHeadsInferred: boolean;
  /** True when headDim came from the `hidden_size / num_attention_heads` fallback. */
  readonly headDimInferred: boolean;
  /** True when an identifiable MoE lacks its expert structure (engine treats it as dense). */
  readonly moeTreatedAsDense: boolean;
  /**
   * True when the MoE active-parameter recomputation was missing
   * `num_attention_heads` and/or `vocab_size`: the missing terms are zeroed,
   * so `activeParamsPerToken` is then a lower bound (the UI shows the
   * "valeurs déduites" badge).
   */
  readonly activeParamsPartial: boolean;
}

// ---------------------------------------------------------------------------
// Typed errors - every HF failure mode is catchable and carries a French
// message the UI can display verbatim.
// ---------------------------------------------------------------------------

/** Base class of every Hugging Face resolution failure. */
export class HfError extends Error {
  readonly name: string = 'HfError';
}

/** HTTP 429 - rate limited; the UI can show a message and retry after a delay. */
export class HfRateLimitError extends HfError {
  readonly name = 'HfRateLimitError';
}

/** The requested model id does not exist on Hugging Face (HTTP 404). */
export class HfModelNotFoundError extends HfError {
  readonly name = 'HfModelNotFoundError';
}

/**
 * Repo without `safetensors.total` (`…-GGUF` and other community
 * quantizations): the total parameter count is not computable.
 */
export class GgufRepoError extends HfError {
  readonly name = 'GgufRepoError';
}

/** HF did not answer within the deadline. */
export class HfTimeoutError extends HfError {
  readonly name = 'HfTimeoutError';
}

/** HF unreachable or protocol-level failure (network error, 5xx, bad payload). */
export class HfUnreachableError extends HfError {
  readonly name = 'HfUnreachableError';
}

/** `config.json` missing, unreadable or incomplete beyond documented fallbacks. */
export class HfConfigError extends HfError {
  readonly name = 'HfConfigError';
}

// ---------------------------------------------------------------------------
// Fetch plumbing
// ---------------------------------------------------------------------------

const RATE_LIMIT_MESSAGE =
  'Limite de requêtes Hugging Face atteinte — réessaie dans un instant';
const TIMEOUT_MESSAGE =
  'Hugging Face ne répond pas (délai dépassé) — vérifie ta connexion';
const UNREACHABLE_MESSAGE = 'Hugging Face injoignable — vérifie ta connexion';
const GGUF_MESSAGE =
  'Impossible de déterminer les paramètres (aucun safetensors) — pointe le modèle de base';
const MALFORMED_PAYLOAD = 'Hugging Face a renvoyé une réponse inattendue';

function isAbort(cause: unknown): boolean {
  return cause instanceof DOMException
    ? cause.name === 'AbortError'
    : cause instanceof Error && cause.name === 'AbortError';
}

/**
 * Fetch wrapper: injects the deadline signal, classifies HTTP failures.
 *
 * 404 mapping is caller-supplied because it is endpoint-specific (a missing
 * model is a `HfModelNotFoundError`, a missing config.json is a
 * `HfConfigError`); 429 and other non-OK statuses are global.
 */
async function hfFetch(
  url: string,
  options: HfRequestOptions,
  errorFor404: (url: string) => HfError,
): Promise<Response> {
  const timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const signal = options.signal
    ? AbortSignal.any([options.signal, AbortSignal.timeout(timeoutMs)])
    : AbortSignal.timeout(timeoutMs);
  let response: Response;
  try {
    response = await fetch(url, { signal });
  } catch (cause) {
    // Caller cancellation is not an HF failure: propagate it untouched so the
    // UI can dismiss aborted autocomplete requests silently.
    if (options.signal?.aborted) throw cause;
    if (isAbort(cause)) throw new HfTimeoutError(TIMEOUT_MESSAGE, { cause });
    throw new HfUnreachableError(UNREACHABLE_MESSAGE, { cause });
  }
  if (response.status === 429) throw new HfRateLimitError(RATE_LIMIT_MESSAGE);
  if (response.status === 404) throw errorFor404(url);
  if (!response.ok) {
    throw new HfUnreachableError(`${UNREACHABLE_MESSAGE} (HTTP ${response.status})`);
  }
  return response;
}

async function readJson(
  response: Response,
  toError: (cause: unknown) => HfError,
): Promise<unknown> {
  try {
    const value: unknown = await response.json();
    return value;
  } catch (cause) {
    throw toError(cause);
  }
}

// ---------------------------------------------------------------------------
// Payload parsing (parse-don't-validate: untrusted JSON crosses once)
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function optionalNumber(
  record: Record<string, unknown>,
  key: string,
): number | undefined {
  const value = record[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function optionalString(
  record: Record<string, unknown>,
  key: string,
): string | undefined {
  const value = record[key];
  return typeof value === 'string' ? value : undefined;
}

/** Short model name: the segment after the org slash of a repo id. */
function shortName(id: string): string {
  const slashIndex = id.lastIndexOf('/');
  return slashIndex === -1 ? id : id.slice(slashIndex + 1);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Autocomplete search over text-generation models.
 * Returns an empty list (no network call) for queries shorter than
 * `SEARCH_MIN_QUERY_LENGTH` characters.
 */
export async function searchModels(
  query: string,
  options: HfRequestOptions = {},
): Promise<readonly ModelSearchHit[]> {
  const trimmed = query.trim();
  if (trimmed.length < SEARCH_MIN_QUERY_LENGTH) return [];
  const url = `${HF_BASE_URL}/api/models?search=${encodeURIComponent(trimmed)}&pipeline_tag=text-generation&limit=${SEARCH_LIMIT}`;
  const response = await hfFetch(url, options, (notFoundUrl) =>
    new HfUnreachableError(`${UNREACHABLE_MESSAGE} (HTTP 404 : ${notFoundUrl})`));
  const payload = await readJson(
    response,
    (cause) => new HfUnreachableError(MALFORMED_PAYLOAD, { cause }),
  );
  if (!Array.isArray(payload)) throw new HfUnreachableError(MALFORMED_PAYLOAD);
  const hits: readonly unknown[] = payload;
  return hits.map((entry) => {
    const id = isRecord(entry) ? optionalString(entry, 'id') : undefined;
    if (id === undefined) throw new HfUnreachableError(MALFORMED_PAYLOAD);
    return { id, name: shortName(id) };
  });
}

/** Head-dim fallback: `hidden_size / num_attention_heads` when both are sane. */
function fallbackHeadDim(
  attentionHeads: number | undefined,
  hiddenSize: number | undefined,
): number | undefined {
  return attentionHeads !== undefined && attentionHeads > 0 && hiddenSize !== undefined
    ? hiddenSize / attentionHeads
    : undefined;
}

/**
 * Maps a model id + `safetensors.total` + a parsed config.json onto the
 * engine `ModelSpec`, with every documented fallback applied and flagged.
 * Throws `HfConfigError` when the config is incomplete beyond the fallbacks.
 */
function toResolvedModel(
  id: string,
  totalParams: number,
  cfg: Record<string, unknown>,
): ResolvedModel {
  const numLayers = optionalNumber(cfg, 'num_hidden_layers');
  if (numLayers === undefined) {
    throw new HfConfigError('config.json incomplet : num_hidden_layers manquant');
  }
  const attentionHeads = optionalNumber(cfg, 'num_attention_heads');
  const kvHeadsDirect = optionalNumber(cfg, 'num_key_value_heads');
  const kvHeads = kvHeadsDirect ?? attentionHeads;
  if (kvHeads === undefined) {
    throw new HfConfigError(
      'config.json incomplet : num_key_value_heads et num_attention_heads manquants',
    );
  }
  const hiddenSize = optionalNumber(cfg, 'hidden_size');
  const headDimDirect = optionalNumber(cfg, 'head_dim');
  const headDim = headDimDirect ?? fallbackHeadDim(attentionHeads, hiddenSize);
  if (headDim === undefined) {
    throw new HfConfigError(
      'config.json incomplet : head_dim et (hidden_size, num_attention_heads) manquants',
    );
  }

  const numExperts =
    optionalNumber(cfg, 'num_experts') ??
    optionalNumber(cfg, 'num_local_experts') ??
    optionalNumber(cfg, 'n_routed_experts');
  const expertsPerTok = optionalNumber(cfg, 'num_experts_per_tok');
  const isMoe =
    numExperts !== undefined && numExperts > 1 &&
    expertsPerTok !== undefined && expertsPerTok > 0;
  const expertIntermediate =
    optionalNumber(cfg, 'moe_intermediate_size') ??
    optionalNumber(cfg, 'intermediate_size');
  const expertFfnSize =
    isMoe && expertIntermediate !== undefined && hiddenSize !== undefined
      ? numLayers * EXPERT_MATRICES * hiddenSize * expertIntermediate
      : undefined;
  // Design §7 : MoE identifiable mais structure d'expert absente -> dense.
  const moeTreatedAsDense = isMoe && expertFfnSize === undefined;

  /**
   * True active parameters per token for a resolvable MoE, stored as
   * `expertSize = A / num_experts_per_tok` so the engine formula
   * (perTok × expertSize) reproduces A:
   *   A = FFN(experts actifs) + attention(2 q_h + 2 kv_h) + embeddings
   *       (+ lm_head si non liés) + routeur
   * Terme absent du config.json → 0 ET flag `activeParamsPartial`
   * (borné inférieure visible, pas de repli caché).
   */
  let expertSize: number | undefined;
  let activeParamsPartial = false;
  if (
    expertFfnSize !== undefined &&
    hiddenSize !== undefined &&
    expertsPerTok !== undefined &&
    expertsPerTok > 0
  ) {
    const attentionParams =
      attentionHeads !== undefined
        ? numLayers * hiddenSize * headDim * (2 * attentionHeads + 2 * kvHeads)
        : 0;
    const vocabSize = optionalNumber(cfg, 'vocab_size');
    const embeddingParams =
      vocabSize !== undefined
        ? vocabSize * hiddenSize * (cfg.tie_word_embeddings === true ? 1 : 2)
        : 0;
    const routerParams =
      numExperts !== undefined ? numLayers * hiddenSize * numExperts : 0;
    activeParamsPartial = attentionHeads === undefined || vocabSize === undefined;
    expertSize =
      (expertsPerTok * expertFfnSize + attentionParams + embeddingParams + routerParams) /
      expertsPerTok;
  }

  const spec: ModelSpec = expertSize !== undefined
    ? {
        totalParams,
        numLayers,
        numKvHeads: kvHeads,
        headDim,
        isMoe: true,
        numExpertsPerTok: expertsPerTok,
        expertSize,
      }
    : { totalParams, numLayers, numKvHeads: kvHeads, headDim, isMoe };

  return {
    ...spec,
    id,
    name: shortName(id),
    activeParamsPerToken: computeActiveParams(spec),
    kvHeadsInferred: kvHeadsDirect === undefined,
    headDimInferred: headDimDirect === undefined,
    moeTreatedAsDense,
    activeParamsPartial,
  };
}

/**
 * Resolves one model from Hugging Face:
 * 1. `GET /api/models/{id}` → `safetensors.total` (N, MoE experts included).
 * 2. `GET {id}/resolve/main/config.json` → architecture fields.
 *
 * Repos without safetensors (`…-GGUF`, AWQ, GPTQ...) throw `GgufRepoError`
 * before any config fetch: the parameter count is not computable, the user
 * must point at the base model.
 */
export async function resolveModel(
  id: string,
  options: HfRequestOptions = {},
): Promise<ResolvedModel> {
  const key = id.trim();
  if (key.length === 0) {
    throw new HfModelNotFoundError('Identifiant de modèle vide');
  }
  const metaUrl = `${HF_BASE_URL}/api/models/${key}`;
  const metaResponse = await hfFetch(metaUrl, options, (notFoundUrl) =>
    new HfModelNotFoundError(`Modèle introuvable sur Hugging Face : ${notFoundUrl}`));
  const metaPayload = await readJson(
    metaResponse,
    (cause) => new HfUnreachableError(MALFORMED_PAYLOAD, { cause }),
  );
  if (!isRecord(metaPayload)) throw new HfUnreachableError(MALFORMED_PAYLOAD);
  const totalParams = isRecord(metaPayload.safetensors)
    ? optionalNumber(metaPayload.safetensors, 'total')
    : undefined;
  if (totalParams === undefined || totalParams <= 0) {
    throw new GgufRepoError(GGUF_MESSAGE);
  }
  const repoId = optionalString(metaPayload, 'id') ?? key;

  const configUrl = `${HF_BASE_URL}/${key}/resolve/main/config.json`;
  const configResponse = await hfFetch(
    configUrl,
    options,
    () => new HfConfigError('config.json introuvable pour ce modèle'),
  );
  const configPayload = await readJson(
    configResponse,
    (cause) => new HfConfigError('config.json illisible pour ce modèle', { cause }),
  );
  if (!isRecord(configPayload)) {
    throw new HfConfigError('config.json illisible pour ce modèle');
  }
  return toResolvedModel(repoId, totalParams, configPayload);
}
