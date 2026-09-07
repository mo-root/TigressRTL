"""Unit tests for src/config.py — defaults, YAML loading, and validation."""

import pytest
import yaml

from config import AgentConfig, DEFAULT_CONFIG_PATH, load_config
from tools import BUILD_TOOL_FUNCTIONS


def write_config(tmp_path, data):
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_none_path_yields_built_in_defaults():
    assert load_config(None) == AgentConfig()


def test_shipped_default_config_loads_and_validates():
    # configs/default.yaml is the file every plain run uses, so a typo in it
    # should fail here rather than at the start of someone's benchmark run.
    assert load_config(DEFAULT_CONFIG_PATH).num_ctx > 0


def test_yaml_values_override_defaults(tmp_path):
    path = write_config(tmp_path, {"model": "llama3.2", "num_ctx": 4096})
    config = load_config(path)
    assert config.model == "llama3.2"
    assert config.num_ctx == 4096


def test_omitted_fields_keep_their_defaults(tmp_path):
    path = write_config(tmp_path, {"model": "llama3.2"})
    defaults = AgentConfig()
    config = load_config(path)
    assert config.num_ctx == defaults.num_ctx
    assert config.max_build_retries == defaults.max_build_retries
    assert config.verilog_build_tool == defaults.verilog_build_tool


def test_empty_yaml_file_is_treated_as_all_defaults(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == AgentConfig()


def test_unknown_key_fails_loudly_instead_of_silently_defaulting(tmp_path):
    # A typo'd key must not silently fall back to a default the user never
    # intended — that would look like the setting simply had no effect.
    path = write_config(tmp_path, {"num_ctx_": 4096})
    with pytest.raises(ValueError, match="Invalid config file"):
        load_config(path)


@pytest.mark.parametrize("backend", sorted(BUILD_TOOL_FUNCTIONS))
def test_every_registered_backend_is_accepted(backend):
    # Keeps config.py's hardcoded validation set in sync with the backends
    # tools.py actually registers, without config.py importing tools.py.
    assert AgentConfig(verilog_build_tool=backend).verilog_build_tool == backend


def test_unknown_build_tool_is_rejected():
    with pytest.raises(ValueError, match="must be 'icarus' or 'slang'"):
        AgentConfig(verilog_build_tool="islang")


def test_unknown_build_tool_from_yaml_is_rejected(tmp_path):
    path = write_config(tmp_path, {"verilog_build_tool": "islang"})
    with pytest.raises(ValueError, match="must be 'icarus' or 'slang'"):
        load_config(path)
