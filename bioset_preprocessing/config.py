"""
Configuration management for the volumetric pipeline.

Supports loading from YAML files, JSON files, or programmatic creation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Union
import logging

logger = logging.getLogger(__name__)


# Available threshold methods
ThresholdMethod = Literal[
    "percentile_95",
    "percentile_90", 
    "percentile_99",
    "otsu",
    "mean_2std",
    "mean_3std",
]


@dataclass
class Config:
    """
    Configuration for the volumetric processing pipeline.
    
    Attributes:
        zarr_url: URL or path to the Zarr store
        zarr_component: Resolution level ("0" = full res, "1" = 2x downsample, etc.)
        channels: List of channel indices to process, or None for all channels
        threshold_method: Thresholding algorithm to use
        threshold_percentile: Percentile value (only used if method is percentile-based)
        tile_size: Size of tiles in Y and X dimensions (auto-calculated if None)
        output_dir: Directory for output files
        save_masks: Whether to save binary masks as TIFFs
        resume: Whether to resume from checkpoint if available
        
    Example:
        >>> config = Config(
        ...     zarr_url="https://example.com/data.zarr",
        ...     channels=[0, 1, 2],
        ...     threshold_method="otsu",
        ... )
    """
    
    zarr_url: str
    zarr_component: str = "0"
    
    channels: Optional[List[int]] = None  # None means all channels
    threshold_method: ThresholdMethod = "percentile_95"
    threshold_percentile: float = 95.0  # Used for percentile methods
    
    # Tiling
    tile_size: Optional[int] = None  # Auto-calculate if None
    
    # Memory settings (for auto tile size calculation)
    available_memory_gb: float = 8.0
    memory_safety_factor: float = 0.5
    
    # Output
    output_dir: Path = field(default_factory=lambda: Path("./results"))
    save_masks: bool = False
    
    # Execution
    resume: bool = True

    # Metadata
    metadata_url: Optional[str] = None  # URL to OME-XML, auto-inferred if None
    
    # Connected Component Filtering (optional)
    cc_filter_enabled: bool = False
    cc_min_volume_um3: float = 0.8  # Minimum component volume in cubic micrometers
    
    # Dilation (optional) - list of radii in micrometers, None means no dilation
    dilation_radii_um: Optional[List[float]] = None  # e.g., [0, 1, 3] or None
    
    # Overlap computation
    max_num_channels_in_comb: int = 4  # Maximum number of channels in a combination
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        valid_methods = [
            "percentile_95", "percentile_90", "percentile_99",
            "otsu", "mean_2std", "mean_3std"
        ]
        if self.threshold_method not in valid_methods:
            raise ValueError(
                f"Invalid threshold_method: {self.threshold_method}. "
                f"Must be one of: {valid_methods}"
            )
        
        if self.threshold_percentile < 0 or self.threshold_percentile > 100:
            raise ValueError("threshold_percentile must be between 0 and 100")
        
        if self.channels is not None:
            if not isinstance(self.channels, list):
                raise ValueError("channels must be a list of integers")
            if len(self.channels) < 2:
                raise ValueError("At least 2 channels required for overlap analysis")
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from a YAML file.
        
        Args:
            path: Path to YAML configuration file
            
        Returns:
            Config instance
            
        Example:
            >>> config = Config.from_yaml("config.yaml")
        """
        import yaml
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {path}")
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Config":
        """
        Load configuration from a JSON file.
        
        Args:
            path: Path to JSON configuration file
            
        Returns:
            Config instance
        """
        import json
        
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(path, "r") as f:
            data = json.load(f)
        
        logger.info(f"Loaded configuration from {path}")
        return cls(**data)
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        import yaml
        
        path = Path(path)
        data = self._to_dict()
        
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved configuration to {path}")
    
    def to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to a JSON file."""
        import json
        
        path = Path(path)
        data = self._to_dict()
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved configuration to {path}")
    
    def _to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "zarr_url": self.zarr_url,
            "zarr_component": self.zarr_component,
            "channels": self.channels,
            "threshold_method": self.threshold_method,
            "threshold_percentile": self.threshold_percentile,
            "tile_size": self.tile_size,
            "available_memory_gb": self.available_memory_gb,
            "memory_safety_factor": self.memory_safety_factor,
            "output_dir": str(self.output_dir),
            "save_masks": self.save_masks,
            "resume": self.resume,
            "metadata_url": self.metadata_url,
            "cc_filter_enabled": self.cc_filter_enabled,
            "cc_min_volume_um3": self.cc_min_volume_um3,
            "dilation_radii_um": self.dilation_radii_um,
        }
    
    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  zarr_url='{self.zarr_url}',\n"
            f"  zarr_component='{self.zarr_component}',\n"
            f"  channels={self.channels},\n"
            f"  threshold_method='{self.threshold_method}',\n"
            f"  tile_size={self.tile_size},\n"
            f"  output_dir='{self.output_dir}',\n"
            f")"
        )
