from pathlib import Path

from pydantic_settings import BaseSettings

from offroad_autonomous_navigator.utils.config_loader import load_settings_from_yaml


class DummySettings(BaseSettings):
    """Dummy class to test configuration loading."""

    host: str = "localhost"
    port: int = 8080
    debug: bool = False


# 1. Load from a YAML file with partial fields: 
# missing fields should use defaults
def test_load_from_yaml_partial_fields(tmp_path: Path) -> None:

    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("host: '192.168.1.100'\ndebug: true\n")

    settings = load_settings_from_yaml(DummySettings, yaml_file)

    assert settings.host == "192.168.1.100"  # Overwritten by YAML
    assert settings.debug is True  # Overwritten by YAML
    assert settings.port == 8080  # Keeps the class default

# 2. Load from a non-existent path: 
# returns an instance with defaults
def test_load_from_non_existent_path(tmp_path: Path) -> None:

    non_existent_file = tmp_path / "missing_config.yaml"

    settings = load_settings_from_yaml(DummySettings, non_existent_file)

    assert settings.host == "localhost"
    assert settings.port == 8080
    assert settings.debug is False

# 3. Load from an empty YAML file: 
# returns an instance with defaults
def test_load_from_empty_yaml(tmp_path: Path) -> None:

    empty_file = tmp_path / "empty.yaml"
    empty_file.write_text("")  # 0-byte file (yaml.safe_load returns None)

    settings = load_settings_from_yaml(DummySettings, empty_file)

    assert settings.host == "localhost"
    assert settings.port == 8080
    assert settings.debug is False