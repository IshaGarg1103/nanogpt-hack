"""Minimal config-file and command-line override support."""

import sys
from ast import literal_eval
from pathlib import Path


for arg in sys.argv[1:]:
    if "=" not in arg:
        assert not arg.startswith("--")
        config_file = Path(arg)
        if not config_file.is_absolute():
            config_file = Path.cwd() / config_file
        print(f"Overriding config with {config_file}:")
        with config_file.open("r", encoding="utf-8") as f:
            print(f.read())
        exec(config_file.read_text(encoding="utf-8"))
    else:
        assert arg.startswith("--")
        key, val = arg.split("=", 1)
        key = key[2:]
        if key not in globals():
            raise ValueError(f"Unknown config key: {key}")
        try:
            attempt = literal_eval(val)
        except (SyntaxError, ValueError):
            attempt = val
        assert type(attempt) is type(globals()[key])
        print(f"Overriding: {key} = {attempt}")
        globals()[key] = attempt
