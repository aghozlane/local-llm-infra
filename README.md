# Dimensionneur de matériel LLM (local-llm-infra)

Estimateur statique, 100 % francophone : à partir d'un **modèle Hugging Face**, d'une
**quantification** et d'une **charge de travail**, il calcule la mémoire VRAM/RAM,
le nombre de GPU (ou l'Apple Silicon adéquat), le TTFT et le débit de décode, puis
propose **trois configurations matérielles** avec leur **budget d'achat en euros**.

## Fonctionnement

1. **Résolution du modèle** — l'API Hugging Face fournit le `config.json`
   (paramètres totaux, params par token, vocabulaire, nombre de couches, GQA,
   quantités de MoE). Les modèles MoE sont détectés ; si la config est incomplète,
   les valeurs déduites sont signalées dans l'interface.
2. **Dimensionnement mémoire** — poids (`bits/param`, KV cache `fp16` ou
   `quant`), overhead de runtime constant, RAM système = modèle + KV + tampon.
   Le poids est réparti sur plusieurs GPU ou sur mémoire unifiée.
3. **Performance** — décode en bande passante (rendu mémoire) avec coefficient
   d'efficacité, préfill en FLOPs avec un coefficient prudent. Ordres de
   grandeur calibrés sur les benchmarks publics de llama.cpp (voir
   [`docs/calibration/`](docs/calibration/)) : viser **±25-30 %** sur décode
   dense ; la VRAM est **jamais sous-estimée** (marge volontaire +25-50 %).
4. **Recommandation** — élimination des configs inadaptées (VRAM, RAM, TTFT,
   débit), tri par prix, et 3 rôles : **la moins chère**, **l'équilibrée** et
   **la plus confortable**, chacune avec sa nomenclature ligne à ligne (GPU,
   plateforme, RAM — prix de catalogues FR versionnés, mis à jour manuellement).

Si aucune config du catalogue ne satisfait les cibles, le site explique pourquoi
et propose des alternatives concrètes (quantifier plus bas, allonger le TTFT,
réduire le débit).

## Développement

```sh
npm install
npm run dev        # serveur local
npm test           # Vitest (moteur, API HF, catalogue, UI format)
npm run typecheck  # tsc -b (strict)
npm run build      # production → dist/
```

## Déploiement

Site statique servi par **GitHub Pages**. Le workflow
[`.github/workflows/deploy.yml`](.github/workflows/deploy.yml) (test, typecheck,
build, déploiement) se déclenche sur `push` vers `main`. Le build utilise des
chemins relatifs (`base: './'`), compatible avec le sous-chemin
`https://<compte>.github.io/<repo>/`.

## Arborescence

| Chemin | Rôle |
|---|---|
| `src/engine/` | Moteur de calcul pur (mémoire, perf, recommandation) — testé unitairement |
| `src/api/` | Client Hugging Face + typage strict du `config.json` |
| `src/data/hardware.json` | Catalogue matériel FR (GPU, plateformes, Apple Silicon, RAM) avec prix |
| `src/components/` | Composants UI React (100 % FR) |
| `src/lib/` | Formatage fr-FR, chargement du catalogue |
| `docs/calibration/` | Note de calibration des constantes face aux benchmarks publics |
| `DESIGN.md` | Système de design (tokens, composants) |

## Limites

- Estimation analytique, **pas un benchmark** : le dimensionnement multi-GPU
  suppose un Tensor Parallel idéalisé ; les exceptions (MoE à court contexte,
  datacenter sur petits modèles) sont affichées dans le bandeau du site.
- Prix = valeurs catalogue au **2026-09-04** (version du catalogue, affichée
  dans le bandeau du site), hors taxes/logistique ; la RAM est facturée au Go
  (meilleur tarif de kits).
