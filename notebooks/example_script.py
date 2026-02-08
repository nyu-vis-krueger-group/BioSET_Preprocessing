"""
Example script for running the BioSET Preprocessing pipeline.
"""

import logging

import ome_types
import requests

from bioset_preprocessing import Config, Pipeline

# Configure logging to see pipeline output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)


def get_unique_channels(metadata_url: str) -> dict[int, str]:
    """Fetch metadata and extract unique channel names."""
    response = requests.get(metadata_url)
    response.raise_for_status()
    ome_xml = ome_types.from_xml(response.text.replace("Â", ""))
    channel_names = [(i, c.name) for i, c in enumerate(ome_xml.images[0].pixels.channels)]

    unique_channels = {}
    for i, name in channel_names:
        if name not in unique_channels.values() and "do not use" not in name:
            unique_channels[i] = name
    return unique_channels


def main():
    metadata_url = "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-invasive-margin/OME/METADATA.ome.xml"
    zarr_url = "https://lsp-public-data.s3.amazonaws.com/biomedvis-challenge-2025/Dataset1-LSP13626-invasive-margin/0"

    unique_channels = get_unique_channels(metadata_url)
    print(f"Found {len(unique_channels)} unique channels: {unique_channels}")

    config = Config(
        zarr_url=zarr_url,
        metadata_url=metadata_url,
        # channels=list(unique_channels.keys()),
        channels=[0,1,2,3,4],
        max_num_channels_in_comb=4,
        enrichment_threshold=2.0,
        tile_size=256,
        zarr_component="0",
        threshold_method="otsu",
        output_dir="./test_results_all",
        cc_filter_enabled=True,
        cc_min_volume_um3=0.8,
        dilation_radii_um=[0, 1, 2, 3, 4, 5],
        hierarchy_levels=4,  # 0,1,2,3
    )

    pipeline = Pipeline(config)
    results = pipeline.run()

    print("\nPipeline completed!")
    if "profiling" in results and "stage_summary" in results["profiling"]:
        print("\nStage Summary:")
        print(results["profiling"]["stage_summary"])

    return results


if __name__ == "__main__":
    main()
