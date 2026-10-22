# ADR-002 — Manifeste de provenance et de lignée des artefacts

- **Statut** : **brouillon de schéma** (J9, 2026-10-22) — non implémenté
- **Décide par** : mission J9, phase 3
- **Contexte** : ADR-001 (disposition), ADR-003 (règle de deux)

## Contexte

Chaque fusion A2D écrit déjà un manifeste `merge_config.json` (α, λ, hashes
d'index des sources, commit git, versions, normes par famille de tenseurs).
Ce manifeste décrit **une** fusion. Il ne décrit pas :

* la **lignée** d'un artefact (quel checkpoint est son parent ?) ;
* le **module producteur** (a2d, legacy, opdlm ?) ;
* la **version** du code qui l'a produit ;
* les **hashes** des fichiers produits (les shards eux-mêmes, pas l'index).

Sans ces informations, un checkpoint fusionné n'est pas auto-descriptif : il
faut retrouver le commit, la commande et les sources dans l'historique.

## Décision (schéma proposé)

Un **manifeste de lignée** par artefact produit, écrit **à côté** de
l'artefact (`lineage.json`), versionné avec le **schéma** :

```jsonc
{
  "schema": "academiia.lineage/1",
  "artifact": {
    "path": "checkpoints/dream_transfer_a09",   // relatif à la racine du dépôt
    "kind": "checkpoint",                        // checkpoint | adapter | eval | report
    "hashes": {                                  // sha256 des fichiers produits
      "model-00001-of-00001.safetensors": "…",
      "model.safetensors.index.json": "…"
    }
  },
  "parent": {
    "path": "checkpoints/dream_7b_base",         // le point de départ (D_b ou D_p)
    "kind": "checkpoint",
    "hashes": { "model.safetensors.index.json": "…" }
  },
  "producer": {
    "module": "a2d",                            // a2d | a2d.legacy | opdlm | core
    "version": "0.1.0",                          // version du paquet
    "commit": "885b143",                         // commit git du producteur
    "command": "modules/a2d/cli/merge_a2d.py --mode transfer --alpha 0.9 …"
  },
  "vector": {                                    // τ, jamais persisté : décrit, pas stocké
    "name": "tau_A",
    "source": "checkpoint_pair",                 // checkpoint_pair | lora_adapter | legacy_npy
    "post": "checkpoints/qwen25_7b_it",
    "base": "checkpoints/qwen25_7b",
    "coefficient": 0.9
  },
  "verification": {
    "post_write_verification": "ok",             // ref. merge_config.json
    "max_ulp": 0
  }
}
```

### Règles

* Le manifeste est **additif** : il ne remplace pas `merge_config.json`, il le
  **référence** (les deux doivent coexister sans contradiction).
* `parent` est **obligatoire** : tout artefact produit a une source déclarée.
  Un artefact sans parent est un artefact *racine* (ex. un checkpoint
  téléchargé) et le dit explicitement (`"parent": null`).
* `producer.module` est **obligatoire** : il permet de refuser un artefact
  produit par un module inconnu (cf. ADR-003).
* Les `hashes` sont des **sha256 de fichier**, jamais des empreintes de
  contenu en mémoire : ils doivent être recalculables par un tiers.

## Conséquences

* Non implémenté en J9 : le schéma est **figé pour la suite**, mais aucun
  code ne l'écrit encore. L'implémentation est un travail distinct (module
  `core` : écriture/lecture du manifeste, hors périmètre J9).
* Le module `opdlm` devra produire le même format de manifeste (ADR-003).
* La provenance reste, en attendant, portée par `merge_config.json` +
  l'historique git.
