from pathlib import Path
from typing import TypeVar

import yaml
from pydantic_settings import BaseSettings

T = TypeVar("T", bound=BaseSettings)

def load_settings_from_yaml(settings_cls: type[T], path: Path) -> T:
    """Load a BaseSettings subclass from a YAML file, 
    falling back to defaults for missing fields."""

    if not path.exists():
        return settings_cls()
    data = yaml.safe_load(path.read_text()) or {} 
    return settings_cls(**data)