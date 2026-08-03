"""Le notebook embarque une copie de python/*.py pour être autonome.

Ces tests garantissent qu'il n'existe qu'une seule source de vérité : si un module
est modifié sans que `make sync-notebook` soit relancé, la copie embarquée devient
obsolète et CE TEST ÉCHOUE -- au lieu de laisser le notebook exécuter du code mort
sur un Colab neuf (où c'est la copie embarquée qui sert).
"""
import json

import pytest

from sync_notebook_fallbacks import (BEGIN, END, MODULES, NOTEBOOK, build_block,
                                     current_block, sync)


@pytest.fixture(scope="module")
def config_cell():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "".join(nb["cells"][1]["source"])


def test_generated_block_is_present(config_cell):
    assert BEGIN in config_cell, "marqueur de début du bloc généré absent"
    assert END in config_cell, "marqueur de fin du bloc généré absent"


def test_embedded_modules_match_their_source(config_cell):
    assert current_block(config_cell) == build_block(), (
        "les modules embarqués dans le notebook ont divergé de python/*.py — "
        "lancez `make sync-notebook`"
    )


def test_check_mode_passes_on_a_synced_notebook():
    assert sync(check_only=True) == 0


@pytest.mark.parametrize("name", list(MODULES))
def test_every_module_is_embedded(config_cell, name):
    assert f"{name!r}:" in config_cell


@pytest.mark.parametrize("name,path", list(MODULES.items()))
def test_embedded_source_is_verbatim(config_cell, name, path):
    source = path.read_text(encoding="utf-8")
    assert source in config_cell, f"{path} n'est pas embarqué mot pour mot"


@pytest.mark.parametrize("name,path", list(MODULES.items()))
def test_source_stays_embeddable(name, path):
    """Garde-fous du quoting de l'embarquement (délimiteur r'''...''')."""
    source = path.read_text(encoding="utf-8")
    assert "'''" not in source, f"{path} contient ''' : impossible à embarquer tel quel"
    assert not source.endswith("\\"), f"{path} finit par un antislash : r-string invalide"


def test_embedded_modules_are_valid_python(config_cell):
    """La copie embarquée doit compiler : sinon le notebook écrirait un module cassé."""
    ns = {}
    start = config_cell.index("_EMBEDDED_MODULES = {")
    end = config_cell.index(END)
    exec(compile(config_cell[start:end], "embedded", "exec"), ns)
    embedded = ns["_EMBEDDED_MODULES"]
    assert set(embedded) == set(MODULES)
    for name, code in embedded.items():
        compile(code, f"{name}.py", "exec")      # lève SyntaxError si cassé
        assert code == MODULES[name].read_text(encoding="utf-8")
