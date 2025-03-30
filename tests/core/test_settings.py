import pytest
import os
import yaml
from pydantic import ValidationError

# Assuming your settings module is in app.core.settings
# Adjust the import path if necessary
from app.core.settings import Settings, get_settings

# Ensure the cache is cleared for tests that modify env vars or files
@pytest.fixture(autouse=True)
def clear_settings_cache():
    get_settings.cache_clear()

@pytest.fixture
def dummy_config_path(tmp_path):
    """Creates a dummy osire_config.yaml in a temporary directory."""
    config_dir = tmp_path / "core"
    config_dir.mkdir()
    config_path = config_dir / "osire_config.yaml"

    dummy_config = {
        "server": {
            "host": "127.0.0.1",
            "port": 9999,
            "web_domain": "http://test.domain",
            "base_url": "/test_prefix"
        },
        "slurm": {
            "output_dir": "/test/slurm/output",
            "preferred_ports": [9000, 9001]
        },
        "ssh": {
            "management_node_count": 1,
            "management_node_pattern": "test-mgmt{}",
            "key_name": "test_key",
            "key_dir": "/test/.ssh"
        },
        "static_models": [
            {
                "id": "test-static-model",
                "model_name": "test/model",
                "engine_type": "test_engine",
                "server_url": "http://test-static:1234"
            }
        ],
        "job_state_manager": {
            "cleanup_interval": 10,
            "update_interval": 5
        },
        "app": {
            "name": "TestApp",
            "version": "1.0.0"
        }
    }

    with open(config_path, 'w') as f:
        yaml.dump(dummy_config, f)

    return config_path


def test_load_settings_from_yaml(dummy_config_path, monkeypatch):
    """Test that settings are correctly loaded from the osire_config.yaml file."""

    # Mock os.path.dirname to point to the directory containing the dummy config
    # This makes Settings._load_yaml_settings find our dummy file
    config_dir = os.path.dirname(dummy_config_path)
    monkeypatch.setattr("app.core.settings.os.path.dirname", lambda x: config_dir)

    # Mock environment variables that are required by Settings
    monkeypatch.setenv("API_TOKEN", "test_token")
    monkeypatch.setenv("SALT", "test_salt")

    # Clear cache before instantiating to ensure fresh load
    get_settings.cache_clear()
    settings = get_settings() # Instantiating Settings triggers loading

    # Assert values from the dummy YAML
    assert settings.server.host == "127.0.0.1"
    assert settings.server.port == 9999
    assert settings.server.web_domain == "http://test.domain"
    # Note: We test base_url override separately
    assert settings.server.base_url_from_config == "/test_prefix"
    assert settings.slurm.output_dir == "/test/slurm/output"
    assert settings.slurm.preferred_ports == [9000, 9001]
    assert settings.ssh.key_name == "test_key"
    assert len(settings.static_models) == 1
    assert settings.static_models[0].id == "test-static-model"
    assert settings.static_models[0].engine_type == "test_engine"
    assert settings.job_state_manager.cleanup_interval == 10
    assert settings.app.name == "TestApp"

    # Assert values from environment variables (which should still be loaded)
    assert settings.API_TOKEN == "test_token"
    assert settings.SALT == "test_salt"

def test_env_variable_override(dummy_config_path, monkeypatch):
    """Test that environment variables override YAML settings."""
    # Mock os.path.dirname to point to the directory containing the dummy config
    config_dir = os.path.dirname(dummy_config_path)
    monkeypatch.setattr("app.core.settings.os.path.dirname", lambda x: config_dir)

    # Mock required environment variables
    monkeypatch.setenv("API_TOKEN", "test_token_env")
    monkeypatch.setenv("SALT", "test_salt_env")

    # Set environment variables to override specific YAML values
    override_port = 8888
    override_slurm_dir = "/env/slurm/override"
    override_app_name = "EnvOverrideApp"

    monkeypatch.setenv("SERVER__PORT", str(override_port))
    monkeypatch.setenv("SLURM__OUTPUT_DIR", override_slurm_dir)
    monkeypatch.setenv("APP__NAME", override_app_name)

    # Clear cache before instantiating
    get_settings.cache_clear()
    settings = get_settings()

    # Assert overridden values
    assert settings.server.port == override_port
    assert settings.slurm.output_dir == override_slurm_dir
    assert settings.app.name == override_app_name
    assert settings.API_TOKEN == "test_token_env"
    assert settings.SALT == "test_salt_env"

    # Assert non-overridden values still come from YAML
    assert settings.server.host == "127.0.0.1" # From dummy_config
    assert settings.job_state_manager.cleanup_interval == 10 # From dummy_config

def test_base_url_override(dummy_config_path, monkeypatch):
    """Test that the BASE_URL env var overrides server.base_url via computed_field."""
    # Mock os.path.dirname
    config_dir = os.path.dirname(dummy_config_path)
    monkeypatch.setattr("app.core.settings.os.path.dirname", lambda x: config_dir)

    # Mock required environment variables
    monkeypatch.setenv("API_TOKEN", "test_token_base")
    monkeypatch.setenv("SALT", "test_salt_base")

    # --- Test with BASE_URL set --- #
    env_base_url = "/from/env/base_url"
    monkeypatch.setenv("BASE_URL", env_base_url)

    get_settings.cache_clear()
    settings_with_env = get_settings()

    # Assert base_url comes from BASE_URL env var
    assert settings_with_env.server.base_url == env_base_url
    # Assert the underlying config value is still loaded correctly
    assert settings_with_env.server.base_url_from_config == "/test_prefix"

    # --- Test without BASE_URL set --- #
    monkeypatch.delenv("BASE_URL", raising=False)

    get_settings.cache_clear()
    settings_without_env = get_settings()

    # Assert base_url falls back to the value from the YAML config
    assert settings_without_env.server.base_url == "/test_prefix"
    assert settings_without_env.server.base_url_from_config == "/test_prefix"

def test_default_values(monkeypatch, tmp_path):
    """Test that default values are used when settings are not in YAML or env."""

    # Simulate no osire_config.yaml by pointing dirname to an empty temp dir
    # (We create a file just so dirname works, but it's not the config file)
    empty_dir = tmp_path / "empty_core"
    empty_dir.mkdir()
    temp_file = empty_dir / "dummy.txt"
    temp_file.touch()
    monkeypatch.setattr("app.core.settings.os.path.dirname", lambda x: str(empty_dir))

    # Mock required environment variables ONLY
    monkeypatch.setenv("API_TOKEN", "default_token")
    monkeypatch.setenv("SALT", "default_salt")

    # Ensure no override env vars are set from previous tests (though fixtures handle this)
    monkeypatch.delenv("SERVER__PORT", raising=False)
    monkeypatch.delenv("BASE_URL", raising=False)

    get_settings.cache_clear()
    settings = get_settings()

    # Assert default values defined in the Settings models
    # Using get_settings() ensures SettingsConfigDict defaults are also applied
    from app.core.settings import ServerConfig, SlurmConfig, SSHConfig, JobStateManagerConfig, AppConfig

    default_server = ServerConfig()
    default_slurm = SlurmConfig()
    default_ssh = SSHConfig()
    default_jsm = JobStateManagerConfig()
    default_app = AppConfig()

    assert settings.server.host == default_server.host # e.g., "0.0.0.0"
    assert settings.server.port == default_server.port # e.g., 8000
    assert settings.server.web_domain == default_server.web_domain # e.g., "https://localhost"
    assert settings.server.base_url == default_server.base_url # e.g., "/"
    assert settings.slurm.output_dir == default_slurm.output_dir
    assert settings.slurm.preferred_ports == default_slurm.preferred_ports
    assert settings.ssh.management_node_count == default_ssh.management_node_count
    assert settings.job_state_manager.cleanup_interval == default_jsm.cleanup_interval
    assert settings.app.name == default_app.name
    assert settings.static_models == [] # Default is empty list

    # Assert required env vars are loaded
    assert settings.API_TOKEN == "default_token"
    assert settings.SALT == "default_salt"

# TODO: Add test for default values when YAML/env vars are missing