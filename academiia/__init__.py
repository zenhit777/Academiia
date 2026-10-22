"""Academiia — espace de recherche A2D (AR-to-Diffusion Transfer).

Deux paquets importables :

* :mod:`academiia.core` — code **générique** (statistiques, empreintes, E/S
  safetensors / fichiers annexes HuggingFace). Il n'importe aucun module.
* :mod:`academiia.a2d` — la méthode A2D (mapping de noms, sources de vecteurs,
  moteur de fusion, adaptateurs). Il importe ``academiia.core``, jamais
  l'inverse.

Règle de dépendance complète : ``docs/adr/ADR-001-disposition.md``.
"""

__version__ = "0.1.0"
