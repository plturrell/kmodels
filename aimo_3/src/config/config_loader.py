"""Configuration loading and management with OmegaConf."""

import re
from pathlib import Path
from typing import Any, Dict, Optional, cast

from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).parent.parent.parent / "configs"


def _ensure_dict(container: Any) -> Dict[str, Any]:
    """
    OmegaConf.to_container() can return non-dict types depending on input.
    We enforce a concrete dict[str, Any] here for strict typing.
    """
    if container is None:
        return {}
    if isinstance(container, dict):
        return cast(Dict[str, Any], container)
    raise TypeError(f"Expected a dict-like config container, got {type(container).__name__}")


class Config:
    """Configuration container."""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize config from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        self._config = OmegaConf.create(config_dict)

    def __getattr__(self, name: str) -> Any:
        """Get configuration value."""
        return getattr(self._config, name)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self._config[key]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._config.get(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return _ensure_dict(OmegaConf.to_container(self._config, resolve=True))

    def merge(self, other: "Config") -> "Config":
        """Merge another config into this one."""
        merged = OmegaConf.merge(self._config, other._config)
        return Config(_ensure_dict(OmegaConf.to_container(merged, resolve=True)))


def load_config(config_name: str, config_dir: Optional[Path] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_name: Configuration name (without .yaml extension)
        config_dir: Directory containing config files (default: configs/)

    Returns:
        Config object
    """
    if config_dir is None:
        config_dir = CONFIG_DIR

    # Prevent path traversal / unexpected config paths.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", config_name):
        raise ValueError(f"Invalid config name: {config_name!r}")

    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_obj = OmegaConf.load(config_path)
    return Config(_ensure_dict(OmegaConf.to_container(config_obj, resolve=True)))


def save_config(config: Config, config_path: Path) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Config object to save
        config_path: Path to save config file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(config=config._config, f=config_path)

