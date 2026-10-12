"""
Check that the heavy dependencies of medunda are not loaded at startup.

Medunda imports its heavy dependencies lazily (see
`medunda.tools.lazy_imports`), so that the command line interface starts
quickly. Ruff (rule TID253) forbids importing these modules at the module
level, but it cannot detect indirect imports, like a type alias evaluated at
runtime that accesses an attribute of a lazy module. Therefore, these tests
run medunda in a new Python process and check which modules have been loaded.
"""

import json
import subprocess
import sys

import pytest

HEAVY_MODULES = (
    "bitsea",
    "cmocean",
    "copernicusmarine",
    "dask",
    "geopandas",
    "matplotlib",
    "netCDF4",
    "numpy",
    "pandas",
    "rioxarray",
    "xarray",
)

IMPORT_SCRIPT = """
import importlib
import json
import sys

importlib.import_module(sys.argv[1])
print(json.dumps(sorted(sys.modules)))
"""

CLI_SCRIPT = """
import json
import runpy
import sys

sys.argv = ["medunda", *sys.argv[1:]]
try:
    runpy.run_module("medunda", run_name="__main__")
except SystemExit:
    pass
print(json.dumps(sorted(sys.modules)))
"""


def _loaded_heavy_modules(script: str, args: list[str]) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        capture_output=True,
        text=True,
        check=True,
    )
    # The list of the modules is always the last line of the output
    loaded_modules = json.loads(result.stdout.splitlines()[-1])
    loaded_packages = {module.split(".")[0] for module in loaded_modules}
    return sorted(loaded_packages.intersection(HEAVY_MODULES))


@pytest.mark.parametrize(
    "module",
    ["medunda", "medunda.__main__", "medunda.tools.xarray_utils"],
)
def test_import_does_not_load_heavy_modules(module):
    assert _loaded_heavy_modules(IMPORT_SCRIPT, [module]) == []


@pytest.mark.parametrize(
    "cli_args",
    [
        ["--help"],
        ["downloader", "--help"],
        ["reducer", "--help"],
        ["plotter", "--help"],
        ["show", "variables"],
        ["show", "providers"],
    ],
)
def test_cli_does_not_load_heavy_modules(cli_args):
    assert _loaded_heavy_modules(CLI_SCRIPT, cli_args) == []
