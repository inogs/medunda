import logging
import yaml
import zipfile
import tempfile
from pathlib import Path
import re

from pydantic import BaseModel
from pydantic import field_validator
import geopandas as gpd


MAIN_DIR = Path(__file__).absolute().parent.parent.parent.parent


LOGGER = logging.getLogger(__name__)


class Domain(BaseModel):
    name: str
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float
    minimum_depth: float | None
    maximum_depth: float | None

    @field_validator('name', mode='after')
    @classmethod
    def name_is_conformal(cls, name: str) -> str:
        """
        Check that the name does not contain special characters.
        If it does, raise a ValueError.
        """
        standard_name = re.compile(r"^[a-zA-Z0-9-]+$")
        if not standard_name.match(name):
            raise ValueError(
                f'The domain name "{name}" contains special characters, '
                "which is not allowed."
            )
        return name


def _read_path(raw_path: str):
    raw_path = raw_path.replace("${MAIN_DIR}", str(MAIN_DIR))
    return Path(raw_path)


def read_zipped_shapefile(compressed_path: Path, temporary_dir:Path):
    """
    Read the content of a zipped file that contains
    only one file (among the others) with an exstension .shp
    and return the path to the uncompressed shapefile.
    """
    with zipfile.ZipFile(compressed_path, 'r') as zip_ref:
        zip_ref.extractall(temporary_dir)

    shapefiles = []
    for f in temporary_dir.glob("**"):
        if not f.is_file():
            continue
        if not f.suffix.lower() == ".shp":
            continue
        shapefiles.append(f)

    if len(shapefiles) == 0:
        raise ValueError(
            f"The file {compressed_path} does not contain a shapefile"
        )
    if len(shapefiles) == 2:
        raise ValueError(
            f"The file {compressed_path} contains more than one shapefile"
        )
    return shapefiles[0]


def read_domain(domain_description: Path) -> Domain:
    yaml_content = domain_description.read_text()
    domain_description_raw = yaml.safe_load(yaml_content)

    # Read the name
    name = domain_description_raw["name"]

    # Read the depth values
    depth = domain_description_raw.get("depth", {})
    min_depth = depth.get("min_depth", None)
    max_depth = depth.get("max_depth", None)

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
        required_dim = [
            "minimum_latitude",
            "maximum_latitude",
            "minimum_longitude",
            "maximum_longitude"]

        ymin = geometry["min_latitude"]
        ymax = geometry["max_latitude"]
        xmin = geometry["min_longitude"]
        xmax = geometry["max_longitude"]

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

    elif geo_type == "shapefile":
        shapefile_path = _read_path(geometry["file_path"])
        if shapefile_path.suffix.lower() == ".zip":
            with tempfile.TemporaryDirectory() as tmp_dir:
                LOGGER.debug("Unzipping file %s", shapefile_path)
                shapefile_path = read_zipped_shapefile(
                    shapefile_path, Path(tmp_dir)
                )

                gdf = gpd.read_file(shapefile_path)
        else:
            gdf = gpd.read_file(shapefile_path)

        # Get the domain from the different ones implemented
        # inside the file
        key_name = geometry["selection_field_name"]
        key_value = geometry["selection_field_value"]
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
