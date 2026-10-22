"""CLI A2D exposes comme points d'entree `console_scripts`.

Les scripts de ce dossier restent executables directement
(`python modules/a2d/cli/merge_a2d.py ...`) : ce `__init__.py` ne fait qu'en
faire un paquet importable (`academiia.a2d.cli`) pour les entry-points
`merge-a2d`, `verify-merge`, `measure-geometry`, `verify-checkpoint`,
`download-models` (J10).
"""
