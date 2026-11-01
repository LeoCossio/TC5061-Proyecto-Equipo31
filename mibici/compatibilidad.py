import sys
import types
from importlib import metadata

from packaging.version import parse


def preparar_deepchecks():
    if "pkg_resources" in sys.modules:
        return

    compat = types.ModuleType("pkg_resources")
    compat.parse_version = parse
    compat.working_set = [
        types.SimpleNamespace(key=d.metadata["Name"].lower())
        for d in metadata.distributions()
        if d.metadata.get("Name")
    ]

    sys.modules["pkg_resources"] = compat