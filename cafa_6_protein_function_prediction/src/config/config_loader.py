"""
Configuration management using OmegaConf.

Provides utilities to load and merge YAML configurations with command-line overrides.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from omegaconf import DictConfig, OmegaConf
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    DictConfig = dict  # Fallback type

LOGGER = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


class Config:
    """Configuration wrapper with attribute access."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary.
        
        Args:
            config_dict: Configuration dictionary
        """
        self._config = config_dict
    
    def __getattr__(self, name: str) -> Any:
        """Get configuration value by attribute access."""
        if name.startswith('_'):
            return object.__getattribute__(self, name)
        return self._config.get(name)
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key access."""
        return self._config[key]
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return dict(self._config)


def load_config(
    config_name: str = "default",
    config_dir: Optional[Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    """Load configuration from YAML file with optional overrides.
    
    Args:
        config_name: Name of config file (without .yaml extension)
        config_dir: Directory containing config files (default: configs/)
        overrides: Dictionary of configuration overrides
    
    Returns:
        Config object
    """
    if not OMEGACONF_AVAILABLE:
        LOGGER.warning("OmegaConf not available, using basic config loading")
        return _load_config_basic(config_name, config_dir, overrides)
    
    config_dir = config_dir or CONFIG_DIR
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load base config
    cfg = OmegaConf.load(config_path)
    
    # Handle defaults (simple inheritance)
    if "defaults" in cfg:
        defaults = cfg.defaults
        if isinstance(defaults, list):
            for default_name in defaults:
                if isinstance(default_name, str) and default_name != config_name:
                    default_cfg = OmegaConf.load(config_dir / f"{default_name}.yaml")
                    cfg = OmegaConf.merge(default_cfg, cfg)
        del cfg["defaults"]
    
    # Apply overrides
    if overrides:
        override_cfg = OmegaConf.create(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    
    # Convert to regular dict for easier access
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    LOGGER.info(f"Loaded configuration from {config_path}")
    return Config(config_dict)


def _load_config_basic(
    config_name: str,
    config_dir: Optional[Path],
    overrides: Optional[Dict[str, Any]],
) -> Config:
    """Basic config loading without OmegaConf (fallback)."""
    import yaml
    
    config_dir = config_dir or CONFIG_DIR
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Apply overrides
    if overrides:
        config_dict.update(overrides)
    
    LOGGER.info(f"Loaded configuration from {config_path} (basic mode)")
    return Config(config_dict)


def save_config(config: Config, output_path: Path) -> None:
    """Save configuration to YAML file.
    
    Args:
        config: Configuration object
        output_path: Path to save configuration
    """
    import yaml
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
    
    LOGGER.info(f"Saved configuration to {output_path}")


__all__ = [
    "Config",
    "load_config",
    "save_config",
    "CONFIG_DIR",
]

