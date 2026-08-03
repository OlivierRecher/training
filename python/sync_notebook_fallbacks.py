"""Embarque les modules du projet dans le notebook, pour qu'il soit autonome.

`train_cellid_llm.ipynb` doit pouvoir tourner sur une machine vierge (ou un Colab
neuf) sans qu'on ait à uploader `python/*.py` à côté. La cellule CONFIG contient
donc un dictionnaire `_EMBEDDED_MODULES` : chaque module y figure en clair, et le
notebook l'écrit sur le disque si l'import échoue.

Ce script régénère ce bloc depuis les fichiers sources. Le risque évident d'une
copie embarquée est qu'elle **divergre** silencieusement de l'original ; c'est
pourquoi `tests/test_notebook_fallbacks.py` compare les deux et échoue si elles ne
correspondent plus. Autrement dit : le notebook reste autonome, mais il n'existe
qu'une seule source de vérité, et la dérive devient une erreur de test.

Usage :
    python python/sync_notebook_fallbacks.py [--check]
    make sync-notebook        # régénère
    make test                 # détecte la dérive
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "train_cellid_llm.ipynb"

# module importable -> fichier source. L'ordre fixe l'ordre d'écriture dans le bloc.
MODULES = {
    "cellid_encoding": ROOT / "python" / "cellid_encoding.py",
    "user_groups": ROOT / "python" / "user_groups.py",
    "generate_sample_users": ROOT / "generate_sample_users.py",
}

BEGIN = "# --- DÉBUT DU BLOC GÉNÉRÉ (python/sync_notebook_fallbacks.py) ---"
END = "# --- FIN DU BLOC GÉNÉRÉ ---"

def build_block():
    """Le bloc `_EMBEDDED_MODULES = {...}` à insérer dans la cellule CONFIG."""
    parts = [BEGIN,
             "# ⚠️  NE PAS ÉDITER À LA MAIN : bloc régénéré par `make sync-notebook`.",
             "# Chaque module de python/ est embarqué ici en clair pour que le notebook soit",
             "# AUTONOME (Colab neuf, machine vierge : aucun fichier annexe à uploader). Le",
             "# notebook n'écrit un module sur le disque que si son import échoue -- une",
             "# installation normale du dépôt continue donc d'utiliser python/<module>.py.",
             "# `tests/test_notebook_fallbacks.py` échoue si une copie ci-dessous diverge de",
             "# son fichier source, pour qu'il n'existe qu'une seule source de vérité.",
             "_EMBEDDED_MODULES = {"]
    for name, path in MODULES.items():
        source = path.read_text(encoding="utf-8")
        # Garde-fous du quoting : le delimiteur ''' ne doit pas apparaître dans la
        # source, et une r-string ne peut pas finir par un antislash.
        if "'''" in source:
            sys.exit(f"ERREUR : {path} contient ''' — impossible à embarquer tel quel "
                     "(changez le délimiteur dans ce script).")
        if source.endswith("\\"):
            sys.exit(f"ERREUR : {path} finit par un antislash — r-string invalide.")
        parts.append(f"    {name!r}: r'''{source}''',")
    parts.append("}")
    parts.append(END)
    return "\n".join(parts) + "\n"


BOOTSTRAP = '''
# Écriture des modules embarqués UNIQUEMENT s'ils ne sont pas importables : sur un
# dépôt normalement installé, python/ est dans sys.path et rien n'est écrit.
_written = []
for _name, _code in _EMBEDDED_MODULES.items():
    try:
        __import__(_name)
    except ImportError:
        Path(f"{_name}.py").write_text(_code, encoding="utf-8")
        _written.append(_name)
if _written:
    import importlib
    importlib.invalidate_caches()
    print("Modules non trouvés — générés automatiquement dans le répertoire courant ✅ : "
          + ", ".join(f"{m}.py" for m in _written))
'''


def current_block(cell_source):
    """Le bloc généré présent dans la cellule, ou None."""
    if BEGIN not in cell_source or END not in cell_source:
        return None
    i = cell_source.index(BEGIN)
    j = cell_source.index(END) + len(END)
    return cell_source[i:j] + "\n"


def sync(check_only=False):
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = nb["cells"][1]
    source = "".join(cell["source"])
    wanted = build_block()
    existing = current_block(source)

    if existing is None:
        if check_only:
            print("ERREUR : bloc généré absent de la cellule CONFIG — lancez "
                  "`make sync-notebook`.")
            return 1
        sys.exit("ERREUR : bloc généré absent ; insérez d'abord les marqueurs "
                 f"{BEGIN!r} / {END!r} dans la cellule CONFIG.")

    if existing == wanted:
        print("Modules embarqués à jour ✅ (" + ", ".join(MODULES) + ")")
        return 0
    if check_only:
        print("ERREUR : les modules embarqués dans le notebook ont divergé de "
              "python/*.py — lancez `make sync-notebook`.")
        return 1

    cell["source"] = source.replace(existing, wanted).splitlines(keepends=True)
    NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8")
    total = sum(len(p.read_text(encoding='utf-8').splitlines()) for p in MODULES.values())
    print(f"Notebook mis à jour : {len(MODULES)} modules embarqués, {total} lignes "
          "(" + ", ".join(MODULES) + ")")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true",
                   help="ne rien écrire ; code de sortie 1 si le notebook a divergé")
    args = p.parse_args()
    raise SystemExit(sync(check_only=args.check))


if __name__ == "__main__":
    main()
