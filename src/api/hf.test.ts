/**
 * Hugging Face client tests (plan step 3) - mocked fetch only, NO network.
 *
 * Fixtures are local config.json / API meta payloads, verbatim from the plan
 * ("Parsing : fixtures config.json locales (aucun réseau dans les tests)").
 */
import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  GgufRepoError,
  HfConfigError,
  HfModelNotFoundError,
  HfRateLimitError,
  HfTimeoutError,
  HfUnreachableError,
  type ResolvedModel,
  resolveModel,
  searchModels,
} from './hf';

const HF = 'https://huggingface.co';

const META_URL = `${HF}/api/models/Qwen/Qwen3-30B-A3B`;
const CONFIG_URL = `${HF}/Qwen/Qwen3-30B-A3B/resolve/main/config.json`;

const QWEN30B_META = {
  id: 'Qwen/Qwen3-30B-A3B',
  safetensors: { total: 30_532_542_464 },
};

const QWEN30B_A3B_CONFIG = {
  num_hidden_layers: 48,
  num_attention_heads: 32,
  num_key_value_heads: 4,
  head_dim: 128,
  hidden_size: 2048,
  vocab_size: 151_936,
  tie_word_embeddings: false,
  moe_intermediate_size: 768,
  num_experts: 128,
  num_experts_per_tok: 8,
};

/** True active params/token of Qwen3-30B-A3B (calibration 2026-09-04, report §7-§8). */
const QWEN_ACTIVE_FFN = 8 * 48 * 3 * 2048 * 768; // active-expert MLP (gate, up, down), all layers
const QWEN_ACTIVE_ATTENTION = 48 * 2048 * 128 * (2 * 32 + 2 * 4); // Q, K, V, O per layer
const QWEN_ACTIVE_EMBEDDINGS = 151_936 * 2048 * 2; // embed + untied lm_head
const QWEN_ACTIVE_ROUTER = 48 * 2048 * 128; // gate to 128 experts, per layer
const QWEN_ACTIVE_PARAMS =
  QWEN_ACTIVE_FFN + QWEN_ACTIVE_ATTENTION + QWEN_ACTIVE_EMBEDDINGS + QWEN_ACTIVE_ROUTER; // 3_352_821_760

function jsonResponse(body: unknown, status: number = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

type RouteMap = Record<string, unknown>;

/** Stubs global fetch with URL-keyed fixture responses (200 by default). */
function stubFetch(routes: RouteMap, status: number = 200) {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const body = routes[String(input)];
    if (body === undefined) {
      return jsonResponse({ error: `URL inattendue : ${String(input)}` }, 404);
    }
    return jsonResponse(body, status);
  });
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

/** Stubs global fetch with a request that never resolves until aborted. */
function stubHangingFetch() {
  const fetchMock = vi.fn(
    (_input: RequestInfo | URL, init?: RequestInit) =>
      new Promise<Response>((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => {
          reject(new DOMException('The operation was aborted.', 'AbortError'));
        });
      }),
  );
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('resolveModel (fetch mocké)', () => {
  it('résout un modèle complet : N, L, kv_h, d_h, MoE actifs, aucun badge déduit', async () => {
    const fetchMock = stubFetch({
      [META_URL]: QWEN30B_META,
      [CONFIG_URL]: QWEN30B_A3B_CONFIG,
    });

    const spec: ResolvedModel = await resolveModel('Qwen/Qwen3-30B-A3B');

    expect(spec.id).toBe('Qwen/Qwen3-30B-A3B');
    expect(spec.name).toBe('Qwen3-30B-A3B');
    expect(spec.totalParams).toBe(30_532_542_464);
    expect(spec.numLayers).toBe(48);
    expect(spec.numKvHeads).toBe(4);
    expect(spec.headDim).toBe(128);
    expect(spec.isMoe).toBe(true);
    expect(spec.numExpertsPerTok).toBe(8);
    expect(spec.expertSize).toBe(QWEN_ACTIVE_PARAMS / 8);
    expect(spec.activeParamsPerToken).toBe(QWEN_ACTIVE_PARAMS); // ≈ 3.35e9, cf. carte HF « 3.3B activated »
    expect(spec.kvHeadsInferred).toBe(false);
    expect(spec.headDimInferred).toBe(false);
    expect(spec.moeTreatedAsDense).toBe(false);
    expect(spec.activeParamsPartial).toBe(false);

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      META_URL,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      CONFIG_URL,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('applique les fallbacks kv_h ← num_attention_heads et d_h ← hidden/heads avec badge « valeurs déduites »', async () => {
    const fetchMock = stubFetch({
      [`${HF}/api/models/org/dense`]: {
        id: 'org/dense',
        safetensors: { total: 8_030_000_000 },
      },
      [`${HF}/org/dense/resolve/main/config.json`]: {
        num_hidden_layers: 32,
        num_attention_heads: 32,
        hidden_size: 4096,
      },
    });

    const spec = await resolveModel('org/dense');

    expect(spec.numKvHeads).toBe(32);
    expect(spec.kvHeadsInferred).toBe(true);
    expect(spec.headDim).toBe(128);
    expect(spec.headDimInferred).toBe(true);
    expect(spec.isMoe).toBe(false);
    expect(spec.moeTreatedAsDense).toBe(false);
    expect(spec.activeParamsPerToken).toBe(8_030_000_000);
    expect(spec.activeParamsPartial).toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('MoE sans vocab_size : A borné inférieurement (sans embeddings) + badge activeParamsPartial', async () => {
    stubFetch({
      [`${HF}/api/models/org/moe-partial`]: {
        id: 'org/moe-partial',
        safetensors: { total: 30_532_542_464 },
      },
      [`${HF}/org/moe-partial/resolve/main/config.json`]: {
        num_hidden_layers: 48,
        num_attention_heads: 32,
        num_key_value_heads: 4,
        head_dim: 128,
        hidden_size: 2048,
        moe_intermediate_size: 768,
        num_experts: 128,
        num_experts_per_tok: 8,
      },
    });

    const spec = await resolveModel('org/moe-partial');

    expect(spec.isMoe).toBe(true);
    expect(spec.moeTreatedAsDense).toBe(false);
    expect(spec.activeParamsPartial).toBe(true);
    // A sans le terme embeddings : FFN + attention + routeur.
    expect(spec.activeParamsPerToken).toBe(QWEN_ACTIVE_FFN + QWEN_ACTIVE_ATTENTION + QWEN_ACTIVE_ROUTER);
  });

  it('traite un MoE sans taille d’expert comme dense + avertissement (design §7)', async () => {
    stubFetch({
      [`${HF}/api/models/org/moe`]: {
        id: 'org/moe',
        safetensors: { total: 30_000_000_000 },
      },
      [`${HF}/org/moe/resolve/main/config.json`]: {
        num_hidden_layers: 48,
        num_attention_heads: 32,
        num_key_value_heads: 4,
        head_dim: 128,
        hidden_size: 2048,
        num_experts: 128,
        num_experts_per_tok: 8,
      },
    });

    const spec = await resolveModel('org/moe');

    expect(spec.isMoe).toBe(true);
    expect(spec.numExpertsPerTok).toBeUndefined();
    expect(spec.expertSize).toBeUndefined();
    expect(spec.moeTreatedAsDense).toBe(true);
    expect(spec.activeParamsPerToken).toBe(spec.totalParams);
  });

  it('échoue avec une erreur typée pour un repo GGUF sans safetensors', async () => {
    const ggufMetaUrl = `${HF}/api/models/org/model-GGUF`;
    const fetchMock = stubFetch({ [ggufMetaUrl]: { id: 'org/model-GGUF' } });

    const promise = resolveModel('org/model-GGUF');
    await expect(promise).rejects.toBeInstanceOf(GgufRepoError);
    await expect(promise).rejects.toThrow(/pointe le modèle de base/);
    // Le config.json ne doit jamais être interrogé quand les paramètres
    // sont indéterminables : un seul appel réseau.
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('échoue avec une erreur typée sur un rate limit HTTP 429', async () => {
    const fetchMock = vi.fn(async () => jsonResponse({ error: 'Too Many Requests' }, 429));
    vi.stubGlobal('fetch', fetchMock);

    await expect(resolveModel('org/x')).rejects.toBeInstanceOf(HfRateLimitError);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it('échoue avec une erreur typée quand HF ne répond pas dans le délai imparti', async () => {
    stubHangingFetch();

    await expect(resolveModel('org/x', { timeoutMs: 20 })).rejects.toBeInstanceOf(HfTimeoutError);
  });

  it('échoue avec une erreur typée quand le modèle est introuvable (404)', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => jsonResponse({ error: 'Not Found' }, 404)),
    );

    await expect(resolveModel('org/inconnu')).rejects.toBeInstanceOf(HfModelNotFoundError);
  });

  it('échoue avec une erreur typée quand config.json est introuvable (404)', async () => {
    const fetchMock = vi.fn(async (input: RequestInfo | URL) =>
      String(input).endsWith('/resolve/main/config.json')
        ? jsonResponse({ error: 'Not Found' }, 404)
        : jsonResponse({ id: 'org/x', safetensors: { total: 1_000_000_000 } }),
    );
    vi.stubGlobal('fetch', fetchMock);

    await expect(resolveModel('org/x')).rejects.toBeInstanceOf(HfConfigError);
  });

  it('échoue avec une erreur typée quand config.json est incomplet au-delà des fallbacks', async () => {
    stubFetch({
      [`${HF}/api/models/org/incomplet`]: {
        id: 'org/incomplet',
        safetensors: { total: 1_000_000_000 },
      },
      [`${HF}/org/incomplet/resolve/main/config.json`]: { num_hidden_layers: 32 },
    });

    await expect(resolveModel('org/incomplet')).rejects.toBeInstanceOf(HfConfigError);
  });

  it('propage l’annulation de l’appelant au lieu de la masquer en erreur HF', async () => {
    const controller = new AbortController();
    controller.abort();
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => {
        throw new DOMException('The operation was aborted.', 'AbortError');
      }),
    );

    await expect(
      resolveModel('org/x', { signal: controller.signal }),
    ).rejects.toBeInstanceOf(DOMException);
  });
});

describe('searchModels (fetch mocké)', () => {
  it('interroge l’API HF (pipeline_tag=text-generation, limite 10), normalise la requête et transmet le signal', async () => {
    const searchUrl = `${HF}/api/models?search=qwen3&pipeline_tag=text-generation&limit=10`;
    const fetchMock = stubFetch({
      [searchUrl]: [{ id: 'Qwen/Qwen3-30B-A3B' }, { id: 'Qwen/Qwen3-32B' }],
    });

    const hits = await searchModels('  qwen3  ');

    expect(hits).toEqual([
      { id: 'Qwen/Qwen3-30B-A3B', name: 'Qwen3-30B-A3B' },
      { id: 'Qwen/Qwen3-32B', name: 'Qwen3-32B' },
    ]);
    // Signal composite (appelant + timeout) : présence seulement, pas d'identité.
    const firstCall = fetchMock.mock.calls[0];
    expect(firstCall).toBeDefined();
    expect(firstCall?.[0]).toBe(searchUrl);
    expect(firstCall?.[1]?.signal).toBeInstanceOf(AbortSignal);
  });

  it('retourne une liste vide sans appel réseau pour une requête trop courte', async () => {
    const fetchMock = stubFetch({});

    expect(await searchModels('ab')).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it('échoue avec une erreur typée si la réponse de recherche est malformée', async () => {
    const searchUrl = `${HF}/api/models?search=qwen3&pipeline_tag=text-generation&limit=10`;
    stubFetch({ [searchUrl]: { error: 'réponse inattendue' } });

    await expect(searchModels('qwen3')).rejects.toBeInstanceOf(HfUnreachableError);
  });
});
