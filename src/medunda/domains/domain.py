import logging
import yaml
from pathlib import Path

from pydantic import BaseModel
import geopandas as gpd


LOGGER = logging.getLogger(__name__)


class Domain(BaseModel):
    name: str
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float
    minimum_depth: float
    maximum_depth: float


GSA9 = Domain(
    name="GSA9",
    minimum_latitude=41.29999921500007,
    maximum_latitude=44.42720294000003,
    minimum_longitude=7.525000098000021,
    maximum_longitude=13.003545339000027,
    minimum_depth=0,
    maximum_depth=800,
)


def _read_path(raw_path: str):
    return Path(raw_path)



def read_domain(domain_description: Path) -> Domain:
    yaml_content = domain_description.read_text()
    domain_description_raw = yaml.safe_load(yaml_content)

    # Read the name
    name = domain_description_raw["name"]

    # Read the depth values
    depth = domain_description_raw["depth"]
    min_depth = depth.get("min_depth", 0)
    max_depth = depth["max_depth"]

    # Read the geometry
    geometry = domain_description_raw["geometry"]
    if "type" not in geometry:
        raise ValueError(
            "No type specified in the geometry of the file."
            )
    
    geo_type = geometry['type'].lower()
    if geo_type not in ("rectangle", "shapefile"):
        raise ValueError(
            f"type must be chosen among rectangle or shapefile;" \
            f"received {geo_type}"
        )

    if geo_type == "rectangle":
        ...
    elif geo_type == "shapefile":
        shapefile_path = _read_path(geometry["file_path"])
        gdf = gpd.read_file(shapefile_path)

        # Get the domain from the different ones implemented
        # inside the file
        key_name = geometry["selection_field_name"]
        key_value = geometry["selection_field_value"]
        print(gdf)
        domain_geometry = gdf.loc[gdf[key_name] == key_value].iloc[0]
    
        xmin, ymin, xmax, ymax = domain_geometry.geometry.bounds

        LOGGER.debug(f"Longitude minimale: {xmin}")
        LOGGER.debug(f"Longitude maximale: {xmax}")
        LOGGER.debug(f"Latitude minimale: {ymin}")
        LOGGER.debug(f"Latitude maximale: {ymax}")
    
        return Domain(
            name=name,
            minimum_latitude=ymin,
            maximum_latitude= ymax,
            minimum_longitude= xmin,
            maximum_longitude= xmax,
            minimum_depth=min_depth,
            maximum_depth=max_depth,
        )

    raise Exception(
        "The geometry type you have chosen should have been implemented "
        "but, because of a bug of the code, it is not recognized"
        )
