"""
OME metadata reader for extracting channel names and physical dimensions.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


@dataclass
class ChannelInfo:
    """Information about a single channel."""
    index: int
    name: str


@dataclass 
class PhysicalDimensions:
    """Physical voxel dimensions in micrometers."""
    x_um: float  # Micrometers per voxel in X
    y_um: float  # Micrometers per voxel in Y
    z_um: float  # Micrometers per voxel in Z
    
    @property
    def voxel_volume_um3(self) -> float:
        """Volume of a single voxel in cubic micrometers."""
        return self.x_um * self.y_um * self.z_um
    
    def __repr__(self) -> str:
        return f"PhysicalDimensions(x={self.x_um}μm, y={self.y_um}μm, z={self.z_um}μm)"


@dataclass
class OMEMetadata:
    """Parsed OME metadata."""
    channels: List[ChannelInfo]
    physical_dimensions: PhysicalDimensions
    
    def get_channel_name(self, index: int) -> str:
        """Get channel name by index."""
        for ch in self.channels:
            if ch.index == index:
                return ch.name
        return f"Channel_{index}"
    
    def get_channel_names(self, indices: List[int]) -> Dict[int, str]:
        """Get channel names for a list of indices."""
        return {i: self.get_channel_name(i) for i in indices}


def load_ome_metadata(
    metadata_url: Optional[str] = None,
    zarr_url: Optional[str] = None,
) -> Optional[OMEMetadata]:
    """
    Load OME metadata from URL.
    
    Args:
        metadata_url: Direct URL to OME-XML file
        zarr_url: Zarr URL (will try to infer metadata location)
        
    Returns:
        OMEMetadata or None if loading fails
    """
    try:
        import requests
        import ome_types
    except ImportError:
        logger.warning("ome_types or requests not installed. Metadata loading disabled.")
        return None
    
    # Determine URL
    url = metadata_url
    if url is None and zarr_url is not None:
        # Try to infer metadata URL from zarr URL
        base_url = zarr_url.rsplit('/', 1)[0]  
        url = f"{base_url}/OME/METADATA.ome.xml"
        logger.debug(f"Inferred metadata URL: {url}")
    
    if url is None:
        logger.warning("No metadata URL provided")
        return None
    
    try:
        logger.info(f"Loading metadata from {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse OME-XML 
        xml_text = response.text.replace("Â", "")
        ome_xml = ome_types.from_xml(xml_text)
        
        # Extract channel info
        channels = [
            ChannelInfo(index=i, name=c.name or f"Channel_{i}")
            for i, c in enumerate(ome_xml.images[0].pixels.channels)
        ]
        
        # Extract physical dimensions
        pixels = ome_xml.images[0].pixels
        physical_dimensions = PhysicalDimensions(
            x_um=float(pixels.physical_size_x or 1.0),
            y_um=float(pixels.physical_size_y or 1.0),
            z_um=float(pixels.physical_size_z or 1.0),
        )
        
        logger.info(f"Loaded metadata: {len(channels)} channels, {physical_dimensions}")
        
        return OMEMetadata(
            channels=channels,
            physical_dimensions=physical_dimensions,
        )
        
    except Exception as e:
        logger.warning(f"Failed to load metadata: {e}")
        return None