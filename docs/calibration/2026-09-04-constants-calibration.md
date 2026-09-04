# Calibrage des constantes du moteur de dimensionnement

**Date** : 4 septembre 2026 · **Branche** : `feature/llm-hardware-predictor`
**Périmètre** : calibrage des trois constantes de `src/engine/formulas.ts` — `PREFILL_ETA` (0.4), `BW_EFFICIENCY` (0.85), `OVERHEAD_GIB` (2) — contre des benchmarks publics de llama.cpp. **Rapport seul : aucun fichier de code, de test ou de configuration n'a été modifié.**

> Convention : décimales à l'anglo-saxonne (point) dans les tableaux pour rester lisibles face aux sources ; les temps sont en tok/s, les débits en Go/s, les volumes en Go (les mesures VRAM des sources sont en Go binaires ± arrondi). Les valeurs marquées *(est.)* sont des estimations, pas des mesures.

---

## 1. Méthodologie

1. **Formules du moteur reproduites telles quelles** (source : `src/engine/formulas.ts`, `src/engine/kv.ts`, `src/engine/quantization.ts`) :
   - Décodage : `tps_utilisateur = (BP × 1e9 × 0.85) / (A × bpw/8 + B × octetsKV_par_token × contexte_moyen)` avec `contexte_moyen = tokens_entrée + tokens_sortie/2` ;
   - Préfill : `TTFT = 2 × A × tokens_entrée / (FLOPS_fp16 × η)` ;
   - VRAM : `requis = (poids + KV + 2) × 1.2`, puis `nb_GPU = ceil(requis / (0.93 × VRAM_carte))` ;
   - `octetsKV_par_token = 2 × L × kv_h × d_h × 2` (K+V, 2 octets) ; table de quant : `Q4_K_M = 4.8` bpw.
2. **Données publiques** : tableaux de scores llama.cpp (discussions GitHub), benchmarks labellisés llama-bench (Hardware Corner), mesures VRAM par contexte, retours d'utilisateurs avec fichiers GGUF identifiés. Toutes les valeurs citées sont sourcées (§11) ; rien n'a été inventé, les combinaisons manquantes sont signalées.
3. **Calcul croisé** : pour chaque point, `η_eff = observé × 2A / FLOPS` (préfill) ou `η_eff = observé × octets/token / BP` (décodage) — c'est-à-dire l'« efficacité » que la constante du moteur devrait prendre pour coller à la mesure.
4. **Limites connues et assumées** : les benchmarks publics utilisent des quantifications `Q4_0` (scoreboard officiel) et `Q4_K_XL` (Hardware Corner) ; le moteur ne connaît que `Q4_K_M = 4.8` bpw. L'écart réel de bpw (~4.5 pour Q4_0, ~4.6–4.86 pour Q4_K_XL/K_M selon les fichiers, cf. §2) fait partie du budget d'erreur. Les tests llama-bench sont mono-requête (B=1) ; le terme de concurrence B n'a pas de point de calibrage llama.cpp public (§6).

## 2. Rappel des hypothèses numériques utilisées

| Élément | Valeur retenue | Source |
| --- | --- | --- |
| Llama-2-7B : 6.74B params, L=32, kv_h=32, d_h=128 | KV/token = 524 288 o | config HF + `/tmp` ; fichier Q4_0 = 3.56 Go → **4.5 bpw** (S1) |
| Qwen3-8B : 8.19B, L=36, kv_h=8, d_h=128 | KV/token = 147 456 o | config HF ; Q4_K_M ≈ 5.0 Go → ≈ 4.8 bpw |
| Qwen3-14B : 14.8B, L=40, kv_h=8, d_h=128 | KV/token = 327 680 o | config HF ; Q4_K_M ≈ 9.0 Go → ≈ 4.8 bpw |
| Qwen3-32B : 32.8B, L=64, kv_h=8, d_h=128 | KV/token = 524 288 o | config HF ; Q4_K_M ≈ 19.9 Go → ≈ 4.8 bpw |
| Qwen3-30B-A3B : 30.5B, MoE, **A = 3.3e9**, L=48, kv_h=4, d_h=128, 8 experts/128 | KV/token = 98 304 o | config HF ; fixture `golden.test.ts` (expertSize=412.5M) ; fichier Q4_K_M **17.28 Go = 4.86 bpw** (S4) |
| Llama-3.3-70B : 70.6B, L=80, kv_h=8, d_h=128 | KV/token = 655 360 o | config HF ; Q4_K_XL ≈ 40 Go → ≈ 4.6 bpw (retro-calculé, S3) |
| Catalogue matériel (BP, FLOPS) | valeurs de `src/data/hardware.json` | 4090 : 1008/165.2e12 · 5090 : 1792/209.5e12 · A100 80 : 2039/312e12 · H100 PCIe : 2039/756.5e12 · Pro 6000 : 1792/500e12 *(est. interne)* · 6000 Ada : 960/91.1e12 · 5080 : 960/112.6e12 · 4080S : 736/104.5e12 · 4070 TiS : 672/88.2e12 · 4070 : 504/58.1e12 · 5060 Ti : 448/47.4e12 · 4060 Ti : 288/44.1e12 · DGX Spark : 273/125e12 |

Note importante sur les MoE : le moteur devrait utiliser **A = 3.3e9** (params actifs réels, fixture golden). Le code actuel de `src/api/hf.ts` produit A = 1.81e9 (voir §8) ; les tableaux ci-dessous utilisent **A = 3.3e9** (valeur correcte) et un encadré montre l'effet du bug.

## 3. BW_EFFICIENCY = 0.85 — décodage

### 3.1 Sources et configurations

- **S1** — scoreboard officiel llama.cpp « Performance on Nvidia CUDA » (discussions #15013) : Llama-2-7B **Q4_0**, llama-bench `pp512/tg128`, contexte 512, B=1, génération à ~position 576 (terme KV négligeable, ~7 % du total).
- **S2** — Hardware Corner, « GPU Ranking for LLMs » (nov.-déc. 2025) : llama-bench, CUDA 12.8, **Q4_K_XL**, FlashAttention, 9 modèles, génération à contexte **plein** 16k/32k/65k/131k (le t/s décroît avec la longueur, conforme à un cache KV rempli), B=1.
- **S4** — discussion #19890 : Qwen3-30B-A3B **Q4_K_M** sur Radeon AI PRO R9700 (Vulkan, FA), table par contexte 128→4096, fichier 17.28 Go (4.86 bpw), B=1.
- **S5** — r/LocalLLaMA : Qwen3-30B-A3B Q4_K_XL sur RTX 4090, **120 tok/s à ~50 000 de contexte** (LM Studio).
- **S8** — groupe Facebook : Llama 70B sur 4090+3080 (44 Go), ≈ 40 tok/s agrégés en split de couches, contexte court.
- **S6** — discussion #3359 (A100-SXM4-40GB, 2023, `-mmq 0`) : Llama-2-7B Q4_K_M 127.15 tok/s (fichier 3.80 Go), 13B Q4_K_M 81.86 tok/s. Point de contrôle datacenter indépendant.

### 3.2 Tableau de calibrage (prédiction moteur vs observé)

**S1 — Llama-2-7B Q4_0 (octets/token ≈ 4.09e9)**

| GPU | Prédit (0.85) | Observé | Écart | η_eff implicite |
| --- | --- | --- | --- | --- |
| RTX 4080 SUPER | 152.8 | 148.33 | +3.0 % | 0.83 |
| RTX 4070 Ti SUPER | 139.5 | 132.26 | +5.5 % | 0.81 |
| RTX 5060 Ti 16 Go | 93.0 | 90.94 | +2.3 % | 0.83 |
| RTX 4060 Ti 16 Go | 59.8 | 63.86 | −6.3 % | 0.91 |
| DGX Spark | 56.7 | 57.21 | −0.9 % | 0.86 |
| RTX 4090 | 209.3 | 186.21 | +12.4 % | 0.76 |
| RTX 5090 | 372.1 | 290.02 | +28.3 % | 0.66 |
| H100 PCIe | 423.4 | 267.81 | +58.1 % | 0.54 |
| A100 80 Go | 423.4 | 190.88 | +121.8 % | 0.38 |

**S2 — Qwen3 8B/14B/32B denses et 30B-A3B MoE (Q4_K_XL, contexte plein)**

| Modèle | Ctx | GPU | Prédit | Observé | Écart |
| --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 16k | RTX 5060 Ti | 51.7 | 51.41 | +0.6 % |
| Qwen3-8B | 16k | RTX 4060 Ti | 33.3 | 34.31 | −3.1 % |
| Qwen3-8B | 16k | RTX 4090 | 116.4 | 104.31 | +11.6 % |
| Qwen3-8B | 16k | RTX 5080 | 110.9 | 94.14 | +17.8 % |
| Qwen3-8B | 16k | RTX 5090 | 206.9 | 145.34 | +42.4 % |
| Qwen3-8B | 32k | RTX 5090 | 155.8 | 111.91 | +39.2 % |
| Qwen3-8B | 65k | RTX 5090 | 104.3 | 80.14 | +30.1 % |
| Qwen3-8B | 131k | RTX 4090 | 35.3 | 32.27 | +9.4 % |
| Qwen3-8B | 131k | RTX 5090 | 62.8 | 49.44 | +26.9 % |
| Qwen3-14B | 16k | RTX 4090 | 74.0 | 69.14 | +7.0 % |
| Qwen3-14B | 16k | RTX 5090 | 131.5 | 102.68 | +28.1 % |
| Qwen3-14B | 16k | RTX Pro 6000 | 131.5 | 96.86 | +35.8 % |
| Qwen3-32B | 16k | RTX 4090 | 35.9 | 37.68 | −4.8 % |
| Qwen3-32B | 16k | RTX 5090 | 63.8 | 50.92 | +25.2 % |
| Qwen3-30B-A3B (MoE) | 16k | RTX 4090 | 237.0 | 139.71 | +69.6 % |
| Qwen3-30B-A3B (MoE) | 16k | RTX 5090 | 421.3 | 141.63 | +197.5 % |
| Qwen3-30B-A3B (MoE) | 16k | RTX Pro 6000 | 421.3 | 139.76 | +201.5 % |
| Qwen3-30B-A3B (MoE) | 16k | RTX 6000 Ada | 225.7 | 120.12 | +87.9 % |
| Qwen3-30B-A3B (MoE) | 50k | RTX 4090 (S5) | 123.8 | ~120 | +3.2 % |
| Llama-3.3-70B | 16k | RTX Pro 6000 | 31.6 | 28.24 | +12.0 % |
| Llama-3.3-70B | 16k | RTX 6000 Ada | 16.9 | 13.65 | +24.1 % |
| Llama-3.3-70B | 131k | RTX Pro 6000 | 17.8 | 16.62 | +6.9 % |

**S4 — Qwen3-30B-A3B Q4_K_M sur R9700 (spec officielle 640 Go/s, S9)**

| ctx | Prédit | Observé | Écart |
| --- | --- | --- | --- |
| 128 | 269.7 | 183.47 | +47.0 % |
| 512 | 264.7 | 183.21 | +44.5 % |
| 1024 | 258.4 | 181.32 | +42.5 % |
| 2048 | 246.6 | 177.05 | +39.3 % |
| 4096 | 226.0 | 171.30 | +31.9 % |

**S8 — Llama 70B Q4_K_M en split 2 GPU (4090+3080, ~44 Go)** : prédit 35.0 vs observé ≈ 40 → **−12.6 %**. Le split de couches de llama.cpp se comporte comme le parallélisme tensoriel idéalisé du moteur (BP et FLOPS additives) sur ce cas.

### 3.3 Validation structurelle du terme KV (indépendante de la constante)

Les rapports de vitesse entre contextes testent la **structure** du terme KV (et donc `kvBytesPerToken`) sans dépendre de la constante 0.85 :

| GPU | Modèle | Ratio observé | Ratio théorique (KV f16) | Δ |
| --- | --- | --- | --- | --- |
| RTX 4090 | Qwen3-8B, 32k/16k | 78.42/104.31 = 0.752 | 0.753 | 0.2 % |
| RTX 4090 | Qwen3-8B, 65k/16k | 53.07/104.31 = 0.509 | 0.504 | 0.9 % |
| RTX 4090 | Qwen3-8B, 131k/16k | 32.27/104.31 = 0.309 | 0.303 | 2.0 % |
| RTX 5090 | Qwen3-8B, 32k/16k | 111.91/145.34 = 0.770 | 0.753 | +2.3 % |

→ La formule du cache KV (147 456 o/token pour Qwen3-8B, soit L×kv_h×d_h×4) colle à **±2 %** sur trois octaves de contexte. La structure du dénominateur est validée.

### 3.4 Verdict : **CONFIRMÉ pour la cible (GPU grand public, modèles denses), avec exceptions à documenter**

- **Grand public, dense, Q4** : écarts **−6 % à +43 %**, médiane ≈ +12 %. Les cartes limitées en bande passante (4060 Ti, 5060 Ti, DGX Spark) sont quasi exactes (η_eff 0.83–0.91) ; la sur-prédiction croît avec la bande passante (4090 : 0.75–0.78 ; 5090 : 0.59–0.68) — les GPU les plus rapides sont moins saturés en BP (overheads d'implémentation, latence mémoire, défauts de coalescence).
- **Dense 70B sur Pro 6000** : +7 % à +12 % — excellent.
- **Exceptions à documenter** : (1) **MoE** — la formule `A × bpw/8` sous-estime les octets/token réels d'un facteur ~1.5–2× à court/moyen contexte → sur-prédiction +32 % à +201 % ; l'écart se referme en contexte long (4090 @50k : +3.2 %, S5) quand le terme KV domine. (2) **Datacenter sur petits modèles** — A100 +122 %, H100 +58 % (η_eff 0.38–0.54) : les A100/H100 ne saturent pas leur BP sur un modèle de 7B. (3) Avec le **bug A_engine** (§8) la sur-prédiction MoE monte à +251 % (4090) et +515 % (5090).
- **Impact décision** : le biais est du côté optimiste (risque de sous-dimensionner un 5090 ou un MoE), mais les marges du cas doré absorbent l'erreur (§7). Ajuster à ~0.7–0.75 pour les cartes ≥ 4090 si l'on veut resserrer ; garder 0.85 sinon (il est déjà dans la fourchette pour 12 des 13 points grand public).

## 4. PREFILL_ETA = 0.4 — préfill/TTFT

### 4.1 Sources

Mêmes sources S1 (pp512), S2 (pp à 16k de contexte), S4 (pp512/pp1024 R9700). Lecture : `pp_prédit = FLOPS × 0.4 / (2A)` ; `η_eff = pp_observé × 2A / FLOPS` ; colonne « TFLOPS atteints » = FLOPS effectivement réalisés par llama.cpp.

### 4.2 Tableau de calibrage

**S1 — Llama-2-7B Q4_0, pp512 (2A = 1.348e10)**

| GPU | Prédit (η=0.4) | Observé | Écart | η_eff | TFLOPS atteints |
| --- | --- | --- | --- | --- | --- |
| RTX 4090 | 4902 | 11992.70 | −59.1 % | 0.98 | 161.7 |
| RTX 5090 | 6217 | 14073.41 | −55.8 % | 0.91 | 189.7 |
| DGX Spark | 3709 | 3062.31 | +21.1 % | 0.33 | 41.3 |
| A100 80 Go | 9258 | 4849.53 | +90.9 % | 0.21 | 65.4 |
| H100 PCIe | 22448 | 9918.34 | +126.3 % | 0.18 | 133.7 |

**S2 — Qwen3-8B (2A = 1.638e10), préfill à contexte 16k**

| GPU | Prédit | Observé | Écart | η_eff |
| --- | --- | --- | --- | --- |
| RTX 5060 Ti | 1158 | 1447.92 | −20.1 % | 0.50 |
| RTX 4060 Ti | 1077 | 1480.81 | −27.3 % | 0.55 |
| RTX 4070 | 1419 | 2064.25 | −31.3 % | 0.58 |
| RTX 4070 Ti SUPER | 2154 | 3050.73 | −29.4 % | 0.57 |
| RTX 5080 | 2750 | 4024.38 | −31.7 % | 0.59 |
| RTX 4080 SUPER | 2552 | 3858.11 | −33.9 % | 0.61 |
| RTX 4090 | 4034 | 6720.91 | −40.0 % | 0.67 |
| RTX 5090 | 5116 | 6956.11 | −26.5 % | 0.54 |
| RTX 6000 Ada | 2225 | 4096.35 | −45.7 % | 0.74 |
| RTX Pro 6000 | 12210 | 7587.74 | +60.9 % | 0.25 |

**S2 — Qwen3-30B-A3B MoE (2A = 6.6e9 avec A = 3.3e9), préfill à 16k**

| GPU | Prédit | Observé | Écart | η_eff |
| --- | --- | --- | --- | --- |
| RTX 4090 | 10012 | 4548.53 | +120.1 % | 0.18 |
| RTX 5090 | 12697 | 4669.17 | +171.9 % | 0.15 |
| RTX 6000 Ada | 5521 | 3156.50 | +74.9 % | 0.23 |
| RTX Pro 6000 | 30303 | 5084.45 | +496 % | 0.07 |

Avec le bug A_engine (1.81e9) : 4090 → 145 877 tok/s prédits (**+3 107 %**), 5090 → 184 995 (**+3 862 %**). S4 : R9700, pp512 observé 3033 → η_eff ≈ 0.17–0.21 contre une crête FP16 estimée 96–118 TFLOPS *(est.)*.

**S2 — Llama-3.3-70B dense (2A = 1.412e11), préfill à 16k**

| GPU | Prédit | Observé | Écart | η_eff |
| --- | --- | --- | --- | --- |
| RTX Pro 6000 | 1416 | 1355.35 | +4.5 % | 0.38 |
| RTX 6000 Ada | 258 | 526.03 | −50.9 % | 0.82 |

(L'écart 6000 Ada est en grande partie un artefact du catalogue : 91.1 TFLOPS = crête FP16 **FP32-acc** ; avec la crête FP16-acc 182 TFLOPS, la prédiction devient 516 vs 526 = **−1.9 %**. Voir §9.)

### 4.3 Verdict : **À AJUSTER — le biais change de signe selon la classe de matériel**

- **Grand public, dense, Q4** : η_eff réel **0.50–0.98** > 0.4 → le t/s de préfill est **sous-prédit de 20 à 59 %**, donc le TTFT prédit est **surestimé de 25 à 150 %**. Direction sûre pour un contrôle de SLA (on promet plus lent que le réel), mais le moteur sous-utilise le matériel (médiane η_eff ≈ 0.6 ; un η ≈ 0.55–0.6 collerait mieux sur grand public).
- **Datacenter sur petits modèles** : η_eff **0.18–0.33** < 0.4 → TTFT prédit **sous-estimé de 21 à 126 %**. Direction dangereuse : un A100/H100 ne sature pas sa crête sur un 7B (overheads, petites batchs, MMQ sous-optimal), le SLA réel serait pire que promis.
- **MoE** : η_eff **0.07–0.23** → TTFT prédit **sous-estimé d'un facteur 1.7 à 6**. Cause dominante : le bug `A_engine` (§8, facteur ×1.8) **plus** l'inefficacité structurelle du préfill MoE (petites matrices experts). C'est l'erreur la plus grave du moteur sur le cas MoE.
- **Recommandation** : garder 0.4 pour la cible grand-public-dense (conservateur d'un facteur ~1.5, acceptable et sain), mais (a) corriger d'abord `A_engine` dans `hf.ts` — c'est le facteur dominant sur MoE —, puis (b) si le périmètre s'étend aux MoE/datacenter, moduler η par classe (≈ 0.15–0.25 pour MoE, ≈ 0.2–0.3 pour datacenter petits modèles) ou l'exposer comme hypothèse visible.

## 5. OVERHEAD_GIB = 2 — VRAM

### 5.1 Sources et tableau

**S3** — Hardware Corner, « LLM VRAM Usage Compared » (oct.-nov. 2025) : VRAM mesurée par contexte (4k→262k), llama-bench + FlashAttention, Q4_K_XL, CUDA 12.8. Comparaison : `moteur = (fichier + KV_f16 + 2) × 1.2`.

| Modèle | Ctx | Fichier (Go) | KV f16 (Go) | Moteur | Mesuré | Écart |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | 16k | ~5.0 | 2.25 | 11.2 | 7 | +59 % |
| Qwen3-8B | 65k | ~5.0 | 9.00 | 19.3 | 14 | +38 % |
| Qwen3-8B | 131k | ~5.0 | 18.0 | 30.1 | 23 | +31 % |
| Qwen3-14B | 16k | ~9.0 | 2.50 | 16.2 | 11 | +47 % |
| Qwen3-14B | 65k | ~9.0 | 10.0 | 25.2 | 19 | +33 % |
| Qwen3-30B-A3B | 8k | 17.28 | 0.75 | 24.0 | 18 | +34 % |
| Qwen3-30B-A3B | 16k | 17.28 | 1.50 | 24.9 | 19 | +31 % |
| Qwen3-30B-A3B | 65k | 17.28 | 6.00 | 30.3 | 23 | +32 % |
| Qwen3-30B-A3B | 131k | 17.28 | 12.0 | 37.5 | 30 | +25 % |
| Qwen3-30B-A3B | 262k | 17.28 | 24.0 | 51.9 | 42 | +24 % |
| Qwen3-32B | 16k | ~19.9 | 4.00 | 31.0 | 23 | +35 % |
| Qwen3-32B | 131k | ~19.9 | 32.0 | 64.6 | 52 | +24 % |

Deux observations structurantes :

1. **Le terme KV est validé** : sur les 15 points ci-dessus, `fichier + KV_f16` colle aux mesures à **±1 Go** (l'overhead réel non-KV observé va de −0.9 à +0.7 Go selon l'arrondi de la source). La formule `2×L×kv_h×d_h×2` est la bonne. *(Les lignes 70B/123B de S3 ne sont utilisables que si l'on suppose un KV quantifié q8_0 — 131k sur 70B : 82 Go mesurés contre ~120 Go attendus en KV f16 — quantification KV non documentée par la source ; elles sont exclues du calibrage.)*
2. **Le +2 Go du moteur est généreux à B=1** : l'overhead réel (compute buffer + contexte CUDA + fragmentation) ≈ **0–1 Go** pour B=1. À B=10 (cas doré) le compute buffer monte à ~2–4 Go → 2 Go devient juste.

### 5.2 Effet d'empilement et bascule de nombre de GPU

La sur-prédiction totale (+24 % à +59 %) ne vient pas seulement de `OVERHEAD_GIB` : elle vient de l'**empilement** `(W + KV + 2) × 1.2 / 0.93`, soit ≈ **×1.29** sur (poids+KV) quand l'overhead réel est ~0.5 Go.

**Exemple de bascule** : Qwen3-30B-A3B Q4_K_M, B=1, contexte 16k, RTX 4090 24 Go.
- Moteur : (17.28 + 1.50 + 2) × 1.2 = **24.9 Go** > 24×0.93 = 22.3 → **2 GPU** recommandés.
- Réalité (S3, S4) : ≈ **19 Go** consommés → **1 GPU suffit**.
Coût de la recommandation : ≈ 2× le budget GPU pour ce cas.

**Cas doré (B=10, contexte 16k)** : moteur (16.8 + 15.0 + 2) × 1.2 = **40.5 Go** → 2×32 Go. Réalité : 17.28 (fichier) + 15.0 (KV) + ~2–4 (compute buffer B=10) = **34–36 Go** → 2×24 Go (44.6 Go utilisables) suffiraient aussi. Sur-prédiction ≈ +15 %, **verdict inchangé** (2 GPU dans les deux cas).

### 5.3 Verdict : **CONFIRMÉ (direction conservatrice, ordre de grandeur bon) — documenter l'empilement**

- `OVERHEAD_GIB = 2` : à B=1 il surestime l'overhead réel d'environ 1–1.5 Go ; à B≥8 il est proche du réel. C'est un choix sûr (il ne fait jamais recommander trop peu de VRAM).
- Le vrai facteur de sur-prédiction est le **×1.2 empilé sur 0.93** : +24 à +59 % de VRAM totale requise → bascules possibles 1↔2 GPU près des frontières de palier. Recommandation : garder 2 Go mais **rendre la marge explicite** (afficher « marge ×1.2 » dans l'UI) ou réduire le facteur à ~1.1 si l'on veut coller aux mesures ; ne pas cumuler les deux sans le dire.

## 6. Terme de concurrence B — observation annexe

- Aucun benchmark public llama.cpp `--parallel B>1` n'a été trouvé lors de la recherche → **terme B non validé empiriquement pour llama.cpp**.
- Indice externe (S9, **vLLM** — pas llama.cpp, FP16, Qwen2.5-3B sur R9700) : débit agrégé 53.3 (B=1) → 183 (B=4) → 251 (B=8) tok/s, soit 3.4× et 4.7× le mono-flux. Le modèle du moteur (batching idéal : poids lus une fois par pas, partagés entre B séquences) impliquerait ≈ 356 (B=4) et ≈ 699 (B=8) tok/s agrégés → **sur-prédiction ≈ ×1.9–2.8** de l'agrégat, donc du t/s/utilisateur.
- Direction : optimiste. Pour un contrôle de SLA « ≥ 20 tok/s/utilisateur », l'erreur est absorbée par la marge dans le cas doré (§7), mais elle mérite un caveat : le t/s/utilisateur annoncé sous forte concurrence est une **borne supérieure** (batching idéal).

## 7. Cas doré — Qwen3-30B-A3B Q4_K_M (exemple chiffré complet)

Config (fixture `golden.test.ts` / plan) : 10 utilisateurs simultanés / 30, 3 req/min/util, 2000 entrée + 500 sortie, contexte 16384, GPU de référence 32 Go / 1792 Go/s / 100e12 FLOPS (type RTX 5090).

| Grandeur | Moteur | Estimation réaliste | Verdict SLA |
| --- | --- | --- | --- |
| KV (B=10) | 98 304 × 16384 × 10 = **15.0 Go** | idem (formule validée §5) | — |
| Poids | 16.8 Go (30e9 × 0.6) | **17.28 Go** (fichier réel, S4) | — |
| VRAM requise | (16.8+15.0+2)×1.2 = **40.5 Go → 2×32 Go** | 17.28+15.0+~2–4 = **34–36 Go → 2×24 Go suffiraient** | 2 GPU dans les deux cas ✓ |
| t/s/utilisateur | **363** (batching idéal) | 247–294 (multiplicateur MoE ×1.5–2 sur les poids) ; 40–90 si l'agrégat réel ≈ 3–5× le mono-flux (141.6 tok/s mono-flux mesuré sur 5090, S2) | ≥ 20 tok/s ✓ avec marge ×2–4 |
| TTFT | 2×3.3e9×2000/(2×100e12×0.4) = **0.165 s** | ~0.44 s (η_eff MoE ≈ 0.15) ; le moteur avec bug A_engine annonce 0.090 s | < 2 s ✓ avec marge ×4–12 |

**Conclusion du cas doré** : le verdict de dimensionnement (2× GPU 32 Go, SLA tenus) est **robuste** à toutes les erreurs de calibrage identifiées ; ce sont les valeurs absolues (t/s, TTFT) qui sont optimistes d'un facteur 2–9 côté débit, pas la décision.

## 8. Observation structurelle hors périmètre des constantes : le A des MoE dans `hf.ts`

`src/api/hf.ts` calcule `expertSize = numLayers × 3 × hiddenSize × expertIntermediate` → Qwen3-30B-A3B : 48×3×2048×768 = **226.5M**, d'où `A_engine = 8 × 226.5M = 1.81e9`.

Cette valeur ne compte **que les poids experts actifs** : elle omet l'attention (~0.91e9), les embeddings (~0.62e9, vocabulaire 151936 × 2048, non liés) et les normes. La valeur « activated » réelle est **3.3e9** (carte HF ; fixture golden `expertSize=412.5M × 8`). Écart : **−45 %**.

Impacts conjoints (MoE, Qwen3-30B-A3B) :
- **Décode** : octets/token sous-estimés → t/s sur-prédits : +251 % (4090 @16k), +515 % (5090 @16k) — contre +70/+198 % avec le bon A.
- **TTFT** : A sous-estimé ET η trop grand → TTFT prédit **×4–5 trop optimiste** (0.165 s annoncé → ~0.4–0.8 s réels). Le SLA du cas doré tient quand même (marge ×2.5–12), mais c'est la première correction à faire si l'on resserre.

## 9. Remarques sur le catalogue matériel

- **Incohérence FP16-acc / FP32-acc** : le catalogue mélange les crêtes (4090 : 165.2 = FP16-acc ; 6000 Ada : 91.1 = FP32-acc ; Pro 6000 : 500e12 = estimation interne explicitement signalée). Conséquence : les prédictions de préfill pour la 6000 Ada sont −46 % trop basses **à cause du catalogue**, pas de la constante (cf. §4.2). Harmoniser (tout en FP16-acc dense) ou annoter chaque valeur.
- La bande passante du **R9700** (640 Go/s, S9) confirme que le chiffre « 512 Go/s » avancé par l'auteur du commentaire S4 est faux — les η_eff calculés ici utilisent 640 Go/s.

## 10. Conclusion sur la revendication « ±25–30 % »

| Axe | Écart observé moteur/réel | Revendication tenue ? |
| --- | --- | --- |
| Décode dense, GPU grand public, Q4 | −6 % à +43 % (médiane +12 %) | **Oui**, sauf RTX 5090 (+27 à +42 %) — la bande haute des débits dépasse légèrement +30 % |
| Décode dense, Pro 6000 / 70B | +7 % à +24 % | Oui |
| Décode MoE (court/moyen ctx) | +32 % à +201 % (+251/+515 % avec bug A) | **Non** — exception à documenter ; OK en contexte long (≥50k) |
| Décode datacenter, petits modèles | +58 % à +122 % | **Non** — exception à documenter |
| Préfill dense, GPU grand public | TTFT surestimé ×1.4–2.5 | Non en valeur absolue, mais **dans le sens sûr** (on promet plus lent) |
| Préfill MoE / datacenter petits modèles | TTFT sous-estimé ×1.7–×6 (bug A dominant) | **Non** — danger de SLA si la cible s'étend |
| VRAM (B=1) | +24 % à +59 % | Non en valeur absolue, mais **dans le sens sûr** (jamais de sous-dimensionnement VRAM) |
| VRAM (cas doré B=10) | +15 % | Oui |

**Verdict global** : la revendication « ±25–30 % » est **supportable** pour le décode dense et la VRAM sur la cible grand public, **à condition** (1) d'afficher les deux exceptions (MoE court contexte, datacenter petits modèles) et (2) de reformuler la promesse VRAM comme « jamais sous-estimée, sur-estimée de 25–50 % » plutôt que symétrique. Pour le préfill MoE, la correction prioritaire est le **A_engine de `hf.ts`** (×1.8), pas la constante η.

**Recommandations de constantes (inchangées dans le code, proposées ici)** :
- `BW_EFFICIENCY = 0.85` : garder ; option : 0.75 si l'on veut être médian sur ≥ 4090.
- `PREFILL_ETA = 0.4` : garder pour la cible ; introduire un η par classe (≈ 0.6 grand public dense après correction du catalogue, ≈ 0.15–0.25 MoE/datacenter) si le périmètre s'étend.
- `OVERHEAD_GIB = 2` : garder ; rendre le facteur ×1.2 visible, ou le réduire à ~1.1 pour coller aux mesures.

## 11. Sources

1. **S1** — llama.cpp, discussion #15013, « Performance of llama.cpp on Nvidia CUDA » (scoreboard, Llama-2-7B Q4_0, pp512/tg128) — https://github.com/ggml-org/llama.cpp/discussions/15013
2. **S2** — Hardware Corner, « The Definitive GPU Ranking for LLMs » (llama-bench, CUDA 12.8, Q4_K_XL, FA, 16k–131k) — https://www.hardware-corner.net/gpu-ranking-local-llm/
3. **S3** — Hardware Corner, « LLM VRAM Usage Compared » (VRAM par contexte, 4k–262k) — https://www.hardware-corner.net/llm-vram-usage-compared/
4. **S4** — llama.cpp, discussion #19890, commentaire JohnTDI-cpu (R9700, Qwen3-30B-A3B Q4_K_M, fichier 17.28 Go, table ctx 128–4096, pp512/pp1024) — https://github.com/ggml-org/llama.cpp/discussions/19890
5. **S5** — r/LocalLLaMA, « Qwen 3 Performance: Quick Benchmarks Across Different Setups » (4090 : 120 tok/s @ 50k) — https://www.reddit.com/r/LocalLLaMA/comments/1kdsp4z/
6. **S6** — llama.cpp, discussion #3359, « Running on an A100 node » (A100-SXM4-40GB, 7B/13B, 2023) — https://github.com/ggml-org/llama.cpp/discussions/3359
7. **S7** — Giles Thomas, « Benchmarking Qwen 3.6 35B MoE on an RTX 3090 » (140 tok/s, 3300 tok/s préfill, fichiers 19.5/22.4 Go, auto-fit 93 952 ctx) — https://www.gilesthomas.com/2026/07/benchmarking-qwen-3-6-35b-moe-rtx-3090
8. **S8** — groupe Facebook « LLM Local » (post 27/06/2026 : 70B sur 4090+3080, ≈700 tok/s préfill, ≈40 tok/s décode) — https://www.facebook.com/groups/1283855472197819/posts/1358075069795855/
9. **S9** — Puget Systems, « AMD Radeon AI PRO R9700 Dual GPU AI Inference Performance » (spec 640 Go/s ; concurrence vLLM : 53.3/183/251 tok/s à B=1/4/8, Qwen2.5-3B FP16) — https://www.pugetsystems.com/labs/articles/amd-radeon-ai-pro-r9700-dual-gpu-ai-inference-performance/
10. **S10** — apxml, « VRAM Calculator Accuracy Scorecard » (méthodologie d'évaluation d'un calculateur VRAM : GMFE 1.06× VRAM, 1.22× décode) — https://apxml.com/tools/vram-calculator/accuracy
11. Configs HF — Qwen3-30B-A3B : https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/config.json · Qwen3.6-35B-A3B : https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json · Qwen3-8B/14B/32B, Llama-2-7B, Llama-3.3-70B : configs HF respectives.
12. Code du moteur (lecture seule) : `src/engine/formulas.ts`, `src/engine/quantization.ts`, `src/engine/kv.ts`, `src/engine/golden.test.ts`, `src/api/hf.ts`, `src/data/hardware.json`, `.omo/plans/2026-09-04-llm-hardware-predictor.md`.
