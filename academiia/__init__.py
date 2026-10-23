"""Academiia — espace de recherche A2D (AR-to-Diffusion Transfer) et d-OPD.

Trois paquets importables :

* :mod:`academiia.core` — code **générique** (statistiques, empreintes, E/S
  safetensors / fichiers annexes HuggingFace). Il n'importe aucun module.
* :mod:`academiia.a2d` — la méthode A2D (mapping de noms, sources de vecteurs,
  moteur de fusion, adaptateurs). Il importe ``academiia.core``, jamais
  l'inverse.
* :mod:`academiia.dopd` — la méthode d-OPD (cible future-aware, configuration,
  contrats d'interface). Il importe ``academiia.core``, jamais ``academiia.a2d``.

Règle de dépendance complète : ``docs/adr/ADR-001-disposition.md``.
"""

__version__ = "0.1.0"
