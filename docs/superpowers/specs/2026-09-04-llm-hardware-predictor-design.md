# Prédicteur de dimensionnement matériel pour LLM local — Design

- **Date** : 2026-09-04
- **Statut** : approuvé en brainstorming — en attente du plan d'implémentation
- **Langue de l'interface** : français uniquement

## 1. Objectif

Un site web **100 % statique** (SPA), déployé sur **GitHub Pages**, qui répond à la question :
« Quel matériel acheter pour servir ce modèle Hugging Face, quantifié ainsi, pour cette charge ? »

À partir d'un modèle HF + d'une quantification + d'un profil de charge, le site calcule et affiche :

- VRAM requise (poids + KV cache + overhead), RAM système, nombre de GPU
- Performances estimées : TTFT, tokens/s par utilisateur, utilisateurs simultanés supportés
- Marge de sécurité
- **Nomenclature achetable (BOM) en euros** : jusqu'à 3 recommandations matérielles (la moins chère / équilibrée / confortable), avec prix unitaires et total

L'outil affiche explicitement ses hypothèses et son ordre de grandeur (±25-30 %) : c'est un **outil d'aide à la décision d'achat**, pas un benchmark.

## 2. Décisions arrêtées en brainstorming

| Sujet | Décision |
|---|---|
| Stack | **Vite + React + TypeScript**, bundle statique, déployé via GitHub Action |
| Catalogue de modèles | **Recherche live via l'API HF uniquement** — pas de liste figée, pas de mode hors-ligne |
| Taille des poids | **Analytique** : paramètres totaux × bits/poids effectifs (pas de lecture des repos GGUF) |
| Catalogue matériel | **Curé, JSON versionné dans le repo**, prix EUR horodatés, mise à jour manuelle par édition + commit |
| Langue UI | **Français uniquement** (pas de couche i18n en v1) |
| Hébergement | **GitHub Pages** (repo créé depuis ce répertoire — nom suggéré : `local-llm-infra`) |

## 3. Architecture

```
┌────────────────────── GitHub Pages (site 100 % statique) ──────────────────────┐
│                                                                                │
│   [UI React FR] ←→ [Moteur de calcul TS pur] ←→ [Catalogue matériel JSON]      │
│        │           (isolé de React, testable)        (versionné, prix EUR)    │
│        │                                                                       │
│        └── fetch HTTPS client-side ──→ API publique Hugging Face (sans clé)   │
│             • GET /api/models?search=…&pipeline_tag=text-generation            │
│             • GET /api/models/{id}            → safetensors.total               │
│             • GET huggingface.co/{id}/raw/main/config.json                     │
└────────────────────────────────────────────────────────────────────────────────┘
```

- **Une seule page** : formulaire à gauche, résultats à droite. Recalcul **en direct** à chaque changement d'entrée (après le premier calcul volontaire).
- **Le moteur de calcul est un module TypeScript pur**, sans dépendance React : entrées typées (`ModelSpec`, `Scenario`, `HardwareCatalog`) → sorties typées (`SizingResult`, `HardwarePick[]`). C'est la seule partie à être testée en profondeur.
- **Aucun backend, aucune clé API.** Les résolutions de modèles sont mises en cache côté client (en mémoire, par ID de modèle, pour la durée de la session).
- **Déploiement** : push sur `main` → GitHub Action (`npm ci` → `npm test` → `vite build` → publication Pages).

## 4. Moteur de calcul (module TS pur)

### 4.1 Résolution du modèle (HF)

- **N** (paramètres totaux) = champ `safetensors.total` de `GET /api/models/{id}`. Pour un MoE, **N compte tous les experts** (tous résidents en mémoire).
- **Architecture** via `config.json` :
  - `L` = `num_hidden_layers`
  - `kv_h` = `num_key_value_heads` (fallback : `num_attention_heads`) — badge « valeurs déduites » si fallback
  - `d_h` = `head_dim` (fallback : `hidden_size / num_attention_heads`)
  - MoE : `num_experts` + `num_experts_per_tok` → **A** (paramètres actifs/token). Si absent → traité comme dense + avertissement.
- Badge MoE dans l'UI, ex. : « MoE détecté : 30 Md totaux, ~3 Md actifs par token ».

### 4.2 Formules

| Grandeur | Formule |
|---|---|
| **Poids (octets)** | `N × bpw / 8` — bpw effectifs fidèles aux GGUF : FP16 = 16, Q8_0 = 8,5, Q6_K = 6,6, Q5_K_M = 5,7, Q4_K_M = 4,8 |
| **KV cache (octets)** | `2 (K+V) × L × kv_h × d_h × 2 (FP16) × contexte_max × séquences_simultanées` |
| **VRAM requise** | `Poids + KV + overhead`, avec `overhead = 2 Go fixes` **puis marge de sécurité** (défaut 20 %) appliquée sur l'ensemble |
| **Nb de GPU** | `ceil(VRAM / VRAM_utilisable_par_GPU)` — tensor parallel implicite : poids et KV répartis. **Borne v1 : ≤ 8 GPU** |
| **Débit décode par utilisateur (tok/s)** | Roofline par pas de décode : `tps_user = BW_eff / (A × bpw/8 + B × kv_octets_par_token × contexte_moyen)` où `B` = séquences simultanées, `contexte_moyen = tokens_entrée + tokens_sortie/2`, `BW_eff ≈ 85 %` de la bande passante catalogue. Débit agrégé = `B × tps_user` (hypothèse *continuous batching*) |
| **TTFT (s)** | Préfill compute-bound : `2 × A × tokens_entrée / (FLOPS_gpu × η)`, `η ≈ 0,4` ; + attente de file si les préfills se sérialisent au-delà de la capacité |
| **Charge offerte (tok/s)** | `utilisateurs_totaux × requêtes_util_min × (tokens_entrée + tokens_sortie) / 60` — comparée au débit agrégé soutenable : la config « absorbe la charge » ou non (✓/✗ affiché) |
| **RAM système** | `max(64 Go, 1,5 × VRAM_totale)` |

**Pourquoi le KV domine** (exemple de validation) : Qwen3-30B-A3B (L=48, kv_h=4, d_h=128) → ~98 Ko de KV **par token et par séquence** → 16 K de contexte × 10 séquences ≈ **16 Go de KV**, contre ~18 Go de poids en Q4_K_M. Le KV cache croît avec le contexte ET le nombre de requêtes simultanées — c'est le point que l'outil doit rendre visible.

### 4.3 Honnêteté & calibration

- Bandeau permanent : « Estimations ±25-30 % — hypothèses : KV en FP16, continuous batching, interconnexion multi-GPU non modélisée au-delà du partage poids/KV ».
- Les constantes (`η`, overhead, `BW_eff`) sont des constantes nommées dans le code, documentées, et **calibrées pendant l'implémentation** contre des benchmarks publics (llama.cpp) — tâche dédiée du plan.

## 5. Catalogue matériel (JSON versionné, prix EUR indicatifs horodatés)

Fichier `src/data/hardware.json`, date de relevé affichée dans l'UI (« prix relevés du 2026-09 »). Les fiches techniques et prix exacts font l'objet d'une **recherche dédiée pendant l'implémentation** (tâche du plan) ; les valeurs ci-dessous sont indicatives.

### 5.1 GPUs (champs : nom, VRAM, VRAM utilisable ~93 %, BW Go/s, FLOPS FP16, TDP W, prix unitaire EUR)

- **Grand public** : RTX 4080 Super, RTX 4090, RTX 5080, RTX 5090
- **Pro** : RTX 6000 Ada 48 Go, RTX PRO 6000 Blackwell 96 Go, L40S 48 Go
- **Datacenter** : A100 80 Go, H100 80 Go, H200 141 Go

### 5.2 Stations IA compactes (mémoire unifiée — plateforme AVEC compute intégré)

Modélisées comme des plateformes avec bande passante et FLOPS propres (pas de VRAM dédiée, pas de multi-unité) :

| Entrée | Mémoire unifiée | BW | Prix indicatif |
|---|---|---|---|
| NVIDIA DGX Spark (GB10) | 128 Go | ~273 Go/s | ~4 500 € |
| AMD Strix Halo — mini-PCs (Ryzen AI Max+ 395 : Framework Desktop, GMKtec EVO-X2, HP Z2 Mini) | 64-128 Go | ~256 Go/s | ~1 800-2 500 € |
| Mac Studio M2 Ultra / M3 Ultra | 96-512 Go | ~800 Go/s | selon config |

**Trade-off à rendre visible** : ces machines font tenir de gros modèles mais plafonnent le débit (BW ~5× plus faible qu'une RTX 5090) → recommandation type « petit budget, usage individuel ou faible simultanéité ».

### 5.3 Plateformes (châssis complets hors GPU : slots max, RAM max, prix EUR)

- Desktop 1-2 GPU : ~2 500 €
- Workstation 2-4 GPU : ~5 000-9 000 €
- Serveur rack 8 GPU : ~25 000 €

### 5.4 RAM

Prix au Go (une valeur, ex. ~5 €/Go DDR5 ECC).

### 5.5 Logique de recommandation

1. Énumérer les combinaisons : chaque GPU × quantité (≤ slots de chaque plateforme), plus les stations compactes.
2. Filtrer : VRAM suffisante **avec marge** ET cibles TTFT/débit respectées.
3. Classer par prix total = `n × GPU + plateforme + RAM`.
4. Présenter jusqu'à **3 recommandations** : **la moins chère**, **équilibrée**, **confortable** (marge de sécurité supérieure).
5. Aucune combinaison viable → écran explicite avec pistes (réduire contexte/simultanés, quantifier davantage).

## 6. Interface (une page, français)

**Formulaire (gauche)** :

1. **Modèle** — recherche HF autocomplétée (debounce ~300 ms, `pipeline_tag=text-generation`), accepte un ID collé (ex. `Qwen/Qwen3-30B-A3B`). Badge MoE / « valeurs déduites » après résolution.
2. **Quantification** — select FP16 / Q8_0 / Q6_K / Q5_K_M / Q4_K_M, affichage immédiat « ≈ X Go de poids ».
3. **Charge** — simultanés, utilisateurs totaux, requêtes/utilisateur/min, tokens entrée moyens, tokens sortie moyens, contexte max (4K → 128K).
4. **Cibles & marge** — TTFT max (s), débit min/utilisateur (tok/s), marge de sécurité % (défaut 20).

**Résultats (droite)** :

1. **Mémoire** — camembert poids / KV / overhead + VRAM totale requise.
2. **Performances** — TTFT et débit estimés, comparés aux cibles (✓/✗), et charge offerte vs capacité agrégée (✓/✗).
3. **Recommandations** — jusqu'à 3 cartes (moins chère / équilibrée / confortable) : config, nb GPU, RAM, prix total EUR, simultanés supportés, tokens/s attendus.
4. **BOM** — nomenclature ligne par ligne, prix unitaires, total (style devis).
5. **Bandeau d'hypothèses** permanent (cf. §4.3).

## 7. Erreurs & cas limites

| Cas | Comportement |
|---|---|
| HF injoignable / timeout | Message clair + bouton « Réessayer ». Pas de fallback caché (choix : tout-live) |
| Repo sans safetensors (quantisations communautaires type `…-GGUF`) | Message : « impossible de déterminer les paramètres — pointe le modèle de **base** » |
| `config.json` incomplet | Fallbacks documentés (§4.1) + badge « valeurs déduites » |
| Rate limit HF (429) | Message + réessai automatique après délai |
| MoE non identifiable | Traité comme dense + avertissement |
| Simultanés < 1 | Clampé à 1 |
| Contexte < tokens entrée + sortie | Avertissement inline |
| Besoin > 8 GPU | « Hors catalogue v1 » + pistes |
| Aucune combinaison viable | Écran explicite (§5.5 étape 5) |

## 8. Tests (vitest — le moteur est pur, aucun navigateur requis)

- **Par formule, paramétré** : poids (FP16/Q8_0/Q6_K/Q5_K_M/Q4_K_M), KV cache (dense, GQA, MoE), VRAM, ceil nb GPU, débit décode, TTFT, RAM.
- **Scénario golden** (validation initiale manuelle, puis non-régression) : `Qwen/Qwen3-30B-A3B` Q4_K_M, 10 simultanés, contexte 16K, 2 000 tokens entrée + 500 sortie, TTFT < 2 s.
- **Parsing** : fixtures `config.json` locales (aucun réseau dans les tests) — cas complet, cas fallback, cas GGUF-only (erreur attendue).
- **Sélection matériel** : catalogues factices → moins chère / équilibrée / confortable, tri, seuils, cas « aucune solution ».
- L'appel réseau HF est simulé (mock) dans les tests du module de résolution.
- **Pas d'E2E navigateur en v1** (YAGNI).

## 9. Hors périmètre v1 (YAGNI assumé)

i18n ; mode hors-ligne / catalogue de modèles figé ; prix temps réel ; tests E2E ; > 8 GPU / multi-châssis ; modélisation de l'interconnexion (NVLink…) au-delà du partage poids/KV ; KV cache quantifié ; speculative decoding ; comparaison de scénarios côte à côte ; comptes utilisateurs / persistance serveur.

## 10. Déploiement & maintenance

- Repo GitHub créé depuis ce répertoire (nom suggéré : `local-llm-infra` — à confirmer avec le compte GitHub de l'utilisateur au moment du déploiement).
- GitHub Action : `npm ci` → `npm test` → `vite build` → publication GitHub Pages.
- **Mise à jour des prix** = éditer `src/data/hardware.json` (prix + date de relevé) + commit. Documenté dans le README.
