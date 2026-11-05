import sys
import types
from importlib import metadata

from packaging.version import parse

from mibici.compatibilidad import preparar_deepchecks

preparar_deepchecks()

# Compatibilidad temporal para DeepChecks 0.19.1
if "pkg_resources" not in sys.modules:
    compat = types.ModuleType("pkg_resources")

    compat.parse_version = parse
    compat.working_set = [
        types.SimpleNamespace(key=d.metadata["Name"].lower())
        for d in metadata.distributions()
        if d.metadata.get("Name")
    ]

    sys.modules["pkg_resources"] = compat