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

/**
 * config.json style Qwen3-VL (multimodal MoE) : l'architecture du décodeur
 * texte vit dans `text_config`, aucun `num_hidden_layers` au premier niveau.
 */
const QWEN3VL_TEXT_CONFIG = {
  model_type: 'qwen3_vl_moe_text',
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
const QWEN3VL_CONFIG = {
  model_type: 'qwen3_vl_moe',
  architectures: ['Qwen3VLMoeForConditionalGeneration'],
  text_config: QWEN3VL_TEXT_CONFIG,
  vision_config: { depth: 27, num_position_embeddings: 2304 },
};

const GGUF_VARIANT_ID = 'Qwen/Qwen3.8-27B-GGUF';
const BASE_ID = 'Qwen/Qwen3.8-27B';
const BASE_META = {
  id: BASE_ID,
  safetensors: { total: 27_400_000_000 },
};
const BASE_CONFIG = {
  num_hidden_layers: 48,
  num_attention_heads: 32,
  num_key_value_heads: 4,
  head_dim: 128,
  hidden_size: 4096,
  vocab_size: 151_936,
};

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
    expect(spec.resolvedFromBaseModel).toBe(false);
    expect(spec.baseModelId).toBeUndefined();
    expect(spec.textConfigUsed).toBeUndefined(); // config texte pur : premier niveau

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

  it('résout un modèle multimodal Qwen3-VL : architecture dans text_config, textConfigUsed', async () => {
    const vlId = 'Qwen/Qwen3-VL-30B-A3B-Instruct';
    stubFetch({
      [`${HF}/api/models/${vlId}`]: {
        id: vlId,
        safetensors: { total: 30_532_542_464 },
      },
      [`${HF}/${vlId}/resolve/main/config.json`]: QWEN3VL_CONFIG,
    });

    const spec: ResolvedModel = await resolveModel(vlId);

    expect(spec.numLayers).toBe(48);
    expect(spec.numKvHeads).toBe(4);
    expect(spec.headDim).toBe(128);
    expect(spec.isMoe).toBe(true);
    expect(spec.numExpertsPerTok).toBe(8);
    expect(spec.expertSize).toBe(QWEN_ACTIVE_PARAMS / 8);
    expect(spec.activeParamsPerToken).toBe(QWEN_ACTIVE_PARAMS);
    expect(spec.totalParams).toBe(30_532_542_464);
    expect(spec.textConfigUsed).toBe(true);
  });

  it('résout un modèle multimodal Qwen3-Omni : architecture dans text_config, textConfigUsed', async () => {
    const omniId = 'Qwen/Qwen3-Omni-30B-A3B-Instruct';
    const omniConfig = {
      model_type: 'qwen3_omni_moe',
      architectures: ['Qwen3OmniMoeForConditionalGeneration'],
      text_config: {
        model_type: 'qwen3_omni_moe_text',
        num_hidden_layers: 36,
        num_attention_heads: 28,
        num_key_value_heads: 4,
        head_dim: 128,
        hidden_size: 2048,
        vocab_size: 151_936,
        tie_word_embeddings: false,
        moe_intermediate_size: 768,
        num_experts: 128,
        num_experts_per_tok: 8,
      },
      audio_config: { num_mel_bins: 128 },
      vision_config: { depth: 27 },
    };
    stubFetch({
      [`${HF}/api/models/${omniId}`]: {
        id: omniId,
        safetensors: { total: 35_000_000_000 },
      },
      [`${HF}/${omniId}/resolve/main/config.json`]: omniConfig,
    });

    const spec: ResolvedModel = await resolveModel(omniId);

    expect(spec.numLayers).toBe(36);
    expect(spec.numKvHeads).toBe(4);
    expect(spec.headDim).toBe(128);
    expect(spec.isMoe).toBe(true);
    expect(spec.textConfigUsed).toBe(true);
  });

  it('préfère num_hidden_layers de premier niveau quand text_config est aussi présent', async () => {
    stubFetch({
      [`${HF}/api/models/org/mixed`]: {
        id: 'org/mixed',
        safetensors: { total: 8_000_000_000 },
      },
      [`${HF}/org/mixed/resolve/main/config.json`]: {
        num_hidden_layers: 32,
        num_attention_heads: 32,
        hidden_size: 4096,
        text_config: {
          num_hidden_layers: 48,
          num_attention_heads: 16,
          hidden_size: 2048,
        },
      },
    });

    const spec: ResolvedModel = await resolveModel('org/mixed');

    expect(spec.numLayers).toBe(32); // premier niveau, pas text_config
    expect(spec.numKvHeads).toBe(32);
    expect(spec.headDim).toBe(128);
    expect(spec.textConfigUsed).toBeUndefined();
  });

  it('préserve textConfigUsed lors du repli sur le modèle de base', async () => {
    const variantId = 'Qwen/Qwen3-VL-30B-A3B-Instruct-GGUF';
    const baseId = 'Qwen/Qwen3-VL-30B-A3B-Instruct';
    stubFetch({
      [`${HF}/api/models/${variantId}`]: {
        id: variantId,
        tags: [`base_model:${baseId}`],
      },
      [`${HF}/api/models/${baseId}`]: {
        id: baseId,
        safetensors: { total: 30_532_542_464 },
      },
      [`${HF}/${baseId}/resolve/main/config.json`]: QWEN3VL_CONFIG,
    });

    const spec: ResolvedModel = await resolveModel(variantId);

    expect(spec.resolvedFromBaseModel).toBe(true);
    expect(spec.baseModelId).toBe(baseId);
    expect(spec.numLayers).toBe(48);
    expect(spec.textConfigUsed).toBe(true);
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

  it('retombe sur le modèle de base pour une variante GGUF (tag base_model), en conservant id/name de la variante', async () => {
    const variantMetaUrl = `${HF}/api/models/${GGUF_VARIANT_ID}`;
    const variantConfigUrl = `${HF}/${GGUF_VARIANT_ID}/resolve/main/config.json`;
    const baseMetaUrl = `${HF}/api/models/${BASE_ID}`;
    const baseConfigUrl = `${HF}/${BASE_ID}/resolve/main/config.json`;

    const fetchMock = stubFetch({
      // safetensors absent : la variante seule est irrésolvable…
      [variantMetaUrl]: {
        id: GGUF_VARIANT_ID,
        tags: [`base_model:${BASE_ID}`],
      },
      // …mais le modèle de base est complet.
      [baseMetaUrl]: BASE_META,
      [baseConfigUrl]: BASE_CONFIG,
    });

    const spec: ResolvedModel = await resolveModel(GGUF_VARIANT_ID);

    expect(spec.id).toBe(GGUF_VARIANT_ID);
    expect(spec.name).toBe('Qwen3.8-27B-GGUF');
    expect(spec.resolvedFromBaseModel).toBe(true);
    expect(spec.baseModelId).toBe(BASE_ID);
    // N et architecture viennent du modèle de base (GGUF même architecture).
    expect(spec.totalParams).toBe(27_400_000_000);
    expect(spec.numLayers).toBe(48);
    expect(spec.numKvHeads).toBe(4);
    expect(spec.headDim).toBe(128);

    // La config de la variante GGUF ne doit jamais être interrogée.
    expect(fetchMock).not.toHaveBeenCalledWith(
      variantConfigUrl,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      baseMetaUrl,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      baseConfigUrl,
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it('retombe sur le modèle de base quand config.json de la variante est incomplet (HfConfigError), en gardant son N', async () => {
    const variantMetaUrl = `${HF}/api/models/org/variant`;
    const variantConfigUrl = `${HF}/org/variant/resolve/main/config.json`;
    const baseMetaUrl = `${HF}/api/models/org/base`;
    const baseConfigUrl = `${HF}/org/base/resolve/main/config.json`;

    const fetchMock = stubFetch({
      // safetensors présents, mais config.json incomplet…
      [variantMetaUrl]: {
        id: 'org/variant',
        safetensors: { total: 8_000_000_000 },
        tags: ['base_model:finetune:org/base'],
      },
      [variantConfigUrl]: { num_hidden_layers: 32 },
      // …le modèle de base fournit l'architecture complète.
      [baseMetaUrl]: { id: 'org/base', safetensors: { total: 8_030_000_000 } },
      [baseConfigUrl]: {
        num_hidden_layers: 32,
        num_attention_heads: 32,
        hidden_size: 4096,
      },
    });

    const spec: ResolvedModel = await resolveModel('org/variant');

    expect(spec.id).toBe('org/variant');
    expect(spec.name).toBe('variant');
    expect(spec.resolvedFromBaseModel).toBe(true);
    expect(spec.baseModelId).toBe('org/base');
    // N = celui de la variante (données propres disponibles) ; l'architecture
    // vient de la config du modèle de base (kv_h et d_h déduits de celle-ci).
    expect(spec.totalParams).toBe(8_000_000_000);
    expect(spec.numLayers).toBe(32);
    expect(spec.numKvHeads).toBe(32);
    expect(spec.kvHeadsInferred).toBe(true);
    expect(spec.headDim).toBe(128);
    expect(spec.headDimInferred).toBe(true);

    expect(fetchMock).toHaveBeenNthCalledWith(1, variantMetaUrl, expect.anything());
    expect(fetchMock).toHaveBeenNthCalledWith(2, variantConfigUrl, expect.anything());
    expect(fetchMock).toHaveBeenNthCalledWith(3, baseMetaUrl, expect.anything());
    expect(fetchMock).toHaveBeenNthCalledWith(4, baseConfigUrl, expect.anything());
  });

  it('échoue avec GgufRepoError pour une variante GGUF sans tag base_model', async () => {
    const ggufMetaUrl = `${HF}/api/models/org/model-GGUF`;
    const fetchMock = stubFetch({
      [ggufMetaUrl]: {
        id: 'org/model-GGUF',
        tags: ['text-generation', 'gguf', 'language:fr'],
      },
    });

    await expect(resolveModel('org/model-GGUF')).rejects.toBeInstanceOf(GgufRepoError);
    // Aucun modèle de base connu : un seul appel réseau, pas de repli caché.
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
      [searchUrl]: [
        { id: 'Qwen/Qwen3-30B-A3B' },
        {
          id: 'Qwen/Qwen3.8-27B-GGUF',
          tags: ['text-generation', 'base_model:quantized:Qwen/Qwen3-30B-A3B'],
        },
        { id: 'Qwen/Qwen3-32B' },
      ],
    });

    const hits = await searchModels('  qwen3  ');

    expect(hits).toEqual([
      { id: 'Qwen/Qwen3-30B-A3B', name: 'Qwen3-30B-A3B' },
      {
        id: 'Qwen/Qwen3.8-27B-GGUF',
        name: 'Qwen3.8-27B-GGUF',
        baseModelId: 'Qwen/Qwen3-30B-A3B',
        baseModelRelation: 'quantized',
      },
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
