from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from types import ModuleType


def import_src_submodule(dotted_path: str) -> ModuleType:
    """Import a module file from the repository `src/` directory by dotted path.

    Example: import_src_submodule('encode_chroma.chroma') will load
    src/encode_chroma/chroma.py as a module and return it.

    This helper avoids importing package-level `__init__` modules during
    pytest collection so tests can remain lightweight and isolated.
    """
    repo_root = Path(__file__).resolve().parents[2]
    rel_path = Path(*dotted_path.split("."))
    module_path = (repo_root / "src" / rel_path).with_suffix(".py")
    spec = spec_from_file_location(dotted_path, str(module_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {module_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
