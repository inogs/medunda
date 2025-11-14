import logging
import re
import tempfile
import zipfile
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Literal
from typing import Union

import geopandas as gpd
import numpy as np
import xarray as xr
import yaml
from bitsea.basins.basin import Basin
from bitsea.basins.region import Polygon
from pydantic import BaseModel
from pydantic import field_validator

MAIN_DIR = Path(__file__).absolute().parent.parent.parent.parent


LOGGER = logging.getLogger(__name__)


class BoundingBox(BaseModel):
    minimum_depth: float | None
    maximum_depth: float | None
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float


class Domain(BaseModel, ABC):
    name: str
    bounding_box: BoundingBox

    @field_validator("name", mode="after")
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

    @abstractmethod
    def compute_selection_mask(self, dataset: xr.Dataset):
        raise NotImplementedError


class RectangularDomain(Domain):
    type: Literal["RectangularDomain"] = "RectangularDomain"

    def compute_selection_mask(self, dataset: xr.Dataset) -> np.ndarray:
        latitudes = dataset.latitude.values
        longitudes = dataset.longitude.values

        mask = np.ones((latitudes.shape[0], longitudes.shape[0]), dtype=bool)
        mask[latitudes < self.bounding_box.minimum_latitude, :] = False
        mask[latitudes > self.bounding_box.maximum_latitude, :] = False
        mask[:, longitudes < self.bounding_box.minimum_longitude] = False
        mask[:, longitudes > self.bounding_box.maximum_longitude] = False

        return mask


class PolygonalDomain(Domain):
    point_latitudes: list[float]
    point_longitudes: list[float]
    type: Literal["PolygonalDomain"] = "PolygonalDomain"

    def compute_selection_mask(self, dataset: xr.Dataset) -> np.ndarray:
        latitudes = dataset.latitude.values
        longitudes = dataset.longitude.values

        polygon = Polygon(
            lat_list=self.point_latitudes, lon_list=self.point_longitudes
        )

        return polygon.is_inside(lon=longitudes, lat=latitudes[:, np.newaxis])

    @classmethod
    def create_from_coordinates(
        cls,
        *,
        name: str,
        longitudes: list[float],
        latitudes: list[float],
        min_depth: float | None = None,
        max_depth: float | None = None,
    ):
        bounding_box = BoundingBox(
            minimum_longitude=min(longitudes),
            maximum_longitude=max(longitudes),
            minimum_latitude=min(latitudes),
            maximum_latitude=max(latitudes),
            minimum_depth=min_depth,
            maximum_depth=max_depth,
        )

        return cls(
            name=name,
            bounding_box=bounding_box,
            point_latitudes=latitudes,
            point_longitudes=longitudes,
        )

    @classmethod
    def create_from_shapely_poly(
        cls,
        name: str,
        poly,
        min_depth: float | None = None,
        max_depth: float | None = None,
    ):
        xx, yy = poly.exterior.coords.xy
        latitudes = yy.tolist()
        longitudes = xx.tolist()

        return cls.create_from_coordinates(
            name=name,
            longitudes=longitudes,
            latitudes=latitudes,
            min_depth=min_depth,
            max_depth=max_depth,
        )


ConcreteDomain = Union[RectangularDomain, PolygonalDomain]


def _read_path(raw_path: str):
    raw_path = raw_path.replace("${MAIN_DIR}", str(MAIN_DIR))
    return Path(raw_path)


def read_zipped_shapefile(compressed_path: Path, temporary_dir: Path):
    """
    Read the content of a zipped file that contains
    only one file (among the others) with an exstension .shp
    and return the path to the uncompressed shapefile.
    """
    with zipfile.ZipFile(compressed_path, "r") as zip_ref:
        LOGGER.debug(
            "Unzipping file %s into %s", compressed_path, temporary_dir
        )
        zip_ref.extractall(temporary_dir)

    shapefiles = []
    for f in temporary_dir.rglob("*"):
        LOGGER.debug("Checking if path %s is a shapefile", f)
        if not f.is_file():
            LOGGER.debug("Skipping path %s because it is not a file", f)
            continue
        if not f.suffix.lower() == ".shp":
            LOGGER.debug("Skipping path %s because it is not a shapefile", f)
            continue
        LOGGER.debug("Found shapefile %s", f)
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


def read_domain(domain_description: Path) -> ConcreteDomain:
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
        raise ValueError("No type specified in the geometry of the file.")

    geo_type = geometry["type"].lower()
    if geo_type not in ("rectangle", "shapefile", "basin", "wkt"):
        raise ValueError(
            f"type must be chosen among rectangle, wkt, basin or shapefile;"
            f"received {geo_type}"
        )

    if geo_type == "rectangle":
        ymin = geometry["min_latitude"]
        ymax = geometry["max_latitude"]
        xmin = geometry["min_longitude"]
        xmax = geometry["max_longitude"]

        LOGGER.debug("Min longitude: %s", xmin)
        LOGGER.debug("Max longitude: %s", xmax)
        LOGGER.debug("Min latitude: %s", ymin)
        LOGGER.debug("Max latitude: %s", ymax)

        bounding_box = BoundingBox(
            minimum_latitude=ymin,
            maximum_latitude=ymax,
            minimum_longitude=xmin,
            maximum_longitude=xmax,
            minimum_depth=min_depth,
            maximum_depth=max_depth,
        )

        return RectangularDomain(name=name, bounding_box=bounding_box)

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

        LOGGER.debug(f"Minimum longitude: {xmin}")
        LOGGER.debug(f"Maximum longitude: {xmax}")
        LOGGER.debug(f"Minimum latitude: {ymin}")
        LOGGER.debug(f"Maximum latitude: {ymax}")

        return PolygonalDomain.create_from_shapely_poly(
            name=name,
            poly=domain_geometry.geometry,
            min_depth=min_depth,
            max_depth=max_depth,
        )

    elif geo_type == "wkt":
        wkt_file = _read_path(geometry["file_path"])
        polygon_name = geometry["polygon_name"]
        with open(wkt_file, "r") as f:
            available_polys = Polygon.read_WKT_file(f)

        try:
            poly = available_polys[polygon_name]
        except KeyError as e:
            available_polys_str = ('"' + pl + '"' for pl in available_polys)
            error_message = (
                f'Polygon "{polygon_name}" not found in {wkt_file}; available '
                f"choices: {', '.join(available_polys_str)}"
            )
            raise KeyError(error_message) from e

        return PolygonalDomain.create_from_coordinates(
            name=name,
            longitudes=poly.border_longitudes,
            latitudes=poly.border_latitudes,
        )
    elif geo_type == "basin":
        basin_uuid = geometry["uuid"]
        basin = Basin.load_from_uuid(basin_uuid)

        if not hasattr(basin, "borders"):
            raise NotImplementedError(
                f"Medunda only supports SimplePolygonalBasins, your current "
                f"basin is a {type(basin)}"
            )
        longitudes = [p[0] for p in basin.borders]
        latitudes = [p[1] for p in basin.borders]

        return PolygonalDomain.create_from_coordinates(
            name=name,
            longitudes=longitudes,
            latitudes=latitudes,
        )

    raise Exception(
        "The geometry type you have chosen should have been implemented "
        "but, because of a bug of the code, it is not recognized"
    )
