import argparse
import logging
import re
import tempfile
import zipfile
from abc import ABC
from abc import abstractmethod
from collections import deque
from pathlib import Path
from typing import Literal
from typing import Union

import yaml
from pydantic import BaseModel
from pydantic import field_validator

import medunda.tools.lazy_imports.bitsea.basin as bitsea_basin
import medunda.tools.lazy_imports.bitsea.region as bitsea_region
from medunda.tools.lazy_imports import geopandas as gpd
from medunda.tools.lazy_imports import numpy as np
from medunda.tools.lazy_imports import xr

MAIN_DIR = Path(__file__).absolute().parent.parent.parent.parent


LOGGER = logging.getLogger(__name__)


class InvalidBasinUUIDError(ValueError):
    pass


class BoundingBox(BaseModel):
    """
    Represents a geographical bounding box with optional depth constraints.

    This class embodies a rectangular geographical region defined by latitude
    and longitude boundaries, optionally including depth constraints.
    The objects of this class are used to define the slices used by the
    providers to select the data from the remote datasets.

    Attributes:
        minimum_depth: Optional lower bound for depth constraint.
        maximum_depth: Optional upper bound for depth constraint.
        minimum_latitude: The southernmost latitude of the bounding box.
        maximum_latitude: The northernmost latitude of the bounding box.
        minimum_longitude: The westernmost longitude of the bounding box.
        maximum_longitude: The easternmost longitude of the bounding box.
    """

    minimum_depth: float | None
    maximum_depth: float | None
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float

    @classmethod
    def merge(cls, *bboxes: "BoundingBox") -> "BoundingBox":
        """Create a new BoundingBox that covers all the given bounding boxes.

        This method returns the smallest bounding box that contains all the
        bounding boxes provided as inputs
        """
        if len(bboxes) == 0:
            raise ValueError("No bounding boxes provided")
        if len(bboxes) == 1:
            return bboxes[0]
        minimum_latitude = min(bbox.minimum_latitude for bbox in bboxes)
        maximum_latitude = max(bbox.maximum_latitude for bbox in bboxes)
        minimum_longitude = min(bbox.minimum_longitude for bbox in bboxes)
        maximum_longitude = max(bbox.maximum_longitude for bbox in bboxes)

        minimum_depth: float | None = None
        maximum_depth: float | None = None

        if all(bbox.minimum_depth is not None for bbox in bboxes):
            minimum_depth = min(bbox.minimum_depth for bbox in bboxes)  # pyright: ignore[reportArgumentType]

        if all(bbox.maximum_depth is not None for bbox in bboxes):
            maximum_depth = max(bbox.maximum_depth for bbox in bboxes)  # pyright: ignore[reportArgumentType]

        return BoundingBox(
            minimum_depth=minimum_depth,
            maximum_depth=maximum_depth,
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
        )


class Domain(BaseModel, ABC):
    """
    Represents a domain in a computational or data-handling context.

    Attributes:
        name (str): The name of the domain. Must conform to the naming
            convention disallowing special characters.
        bounding_box (BoundingBox): The bounding box representing the spatial
            boundaries of the domain.

    Methods:
        name_is_conformal(name: str) -> str:
            Validates that the domain name complies with the naming standard by
            disallowing special characters. Returns the name if valid.

        compute_selection_mask(dataset: xr.Dataset) -> np.ndarray:
            Abstract method to compute a selection mask over a dataset.
            Must be implemented by subclasses.
    """

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
    def compute_selection_mask(self, dataset: "xr.Dataset") -> "np.ndarray":
        raise NotImplementedError


class RectangularDomain(Domain):
    type: Literal["RectangularDomain"] = "RectangularDomain"

    def compute_selection_mask(self, dataset: "xr.Dataset") -> "np.ndarray":
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

    def compute_selection_mask(self, dataset: "xr.Dataset") -> "np.ndarray":
        latitudes = dataset.latitude.values
        longitudes = dataset.longitude.values

        polygon = bitsea_region.Polygon(
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


class MultiPolygonalDomain(Domain):
    type: Literal["MultiPolygonalDomain"] = "MultiPolygonalDomain"
    polygons: list[PolygonalDomain]

    def compute_selection_mask(self, dataset: "xr.Dataset") -> "np.ndarray":
        latitudes = dataset.latitude.values[:, np.newaxis]
        longitudes = dataset.longitude.values

        data_shape = np.broadcast_shapes(latitudes.shape, longitudes.shape)
        mask = np.zeros(data_shape, dtype=bool)

        if len(self.polygons) == 0:
            return mask

        for polygonal_domain in self.polygons:
            poly_mask = polygonal_domain.compute_selection_mask(
                dataset=dataset
            )
            mask = np.logical_or(mask, poly_mask)

        return mask

    @classmethod
    def create_from_polygons(cls, name: str, polygons: list[PolygonalDomain]):
        bounding_box = BoundingBox.merge(*[p.bounding_box for p in polygons])
        return cls(name=name, bounding_box=bounding_box, polygons=polygons)


ConcreteDomain = Union[
    RectangularDomain, PolygonalDomain, MultiPolygonalDomain
]


def _read_path(raw_path: str):
    raw_path = raw_path.replace("${MAIN_DIR}", str(MAIN_DIR))
    return Path(raw_path)


def domain_from_basin(basin_uuid: str, name: str | None = None):
    """
    Generates a domain representation from a given basin identifier.

    This function constructs a domain representation from a basin specified by
    its UUID. If the basin is a composed entity, it processes its components
    recursively to generate the appropriate domain representation. The result
    could be either a simple polygonal domain or a multipolygonal domain,
    depending on the type of the basin. It ensures that unsupported basin
    types are appropriately handled by raising an error.

    Parameters:
        basin_uuid (str): The unique identifier for the basin to process.
        name (str | None, optional): The optional name to assign to the
            created domain. If not provided, the name is derived from the
            basin UUID.

    Returns:
        PolygonalDomain | MultiPolygonalDomain: The created domain
            representing the given basin. Returns a PolygonalDomain for
            simple basins, and a MultiPolygonalDomain for composed basins.

    Raises:
        InvalidBasinUUIDError: If the provided basin UUID is invalid
        ValueError: If processing a composed basin that has no sub-basins.
        NotImplementedError: If encountering a basin type that is not
            supported (only SimplePolygonalBasins are supported).
    """
    LOGGER.debug("Loading basin %s", basin_uuid)
    try:
        basin = bitsea_basin.Basin.load_from_uuid(basin_uuid)
    except Exception as e:
        raise InvalidBasinUUIDError(
            f"Could not load basin {basin_uuid}"
        ) from e

    if name is None:
        name = basin_uuid.split(".")[-1]

    to_be_checked = deque([basin])
    simple_basins = {}

    # If the basin is composed, add its components to the list of simple basins
    # that we have to check
    while len(to_be_checked) > 0:
        current_basin = to_be_checked.pop()
        basin_uuid = current_basin.get_uuid()
        LOGGER.debug("Checking if basin %s is a composed basin", basin_uuid)
        if basin_uuid in simple_basins:
            LOGGER.debug(
                "Basin %s has already been visited! Skipped", basin_uuid
            )
            continue
        if hasattr(current_basin, "basin_list"):
            LOGGER.debug(
                "Basin %s is composed! Adding its components to the list of "
                "basins that we have to check",
                basin_uuid,
            )
            for basin_component in reversed(current_basin.basin_list):
                LOGGER.debug(
                    "Adding basin %s to the list of basins that we have to check",
                    basin_component.get_uuid(),
                )
                to_be_checked.append(basin_component)
        else:
            LOGGER.debug(
                "Basin %s is not composed! Adding it to the list of simple basins",
                basin_uuid,
            )
            simple_basins[basin_uuid] = current_basin

    if len(simple_basins) == 0:
        raise ValueError(
            f"The basin {basin_uuid} is a composed basin but it does not "
            "contain any other basin"
        )

    if len(simple_basins) == 1:
        basin = list(simple_basins.values())[0]
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

    polygons = []
    for current_basin in simple_basins.values():
        if not hasattr(current_basin, "borders"):
            raise NotImplementedError(
                f"Medunda only supports SimplePolygonalBasins, but basin "
                f"{current_basin.name} (that is part of your composed basin "
                f"{basin.name}) is a {type(current_basin)}"
            )
        longitudes = [p[0] for p in current_basin.borders]
        latitudes = [p[1] for p in current_basin.borders]

        current_polygon = PolygonalDomain.create_from_coordinates(
            name=current_basin.name,
            longitudes=longitudes,
            latitudes=latitudes,
        )
        polygons.append(current_polygon)

    return MultiPolygonalDomain.create_from_polygons(
        name=name, polygons=polygons
    )


def read_zipped_shapefile(compressed_path: Path, temporary_dir: Path):
    """
    Read the content of a zipped file that contains
    only one file (among the others) with an extension .shp
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


def read_domain_from_yaml(yaml_path: Path) -> ConcreteDomain:
    yaml_content = yaml_path.read_text()
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
            available_polys = bitsea_region.Polygon.read_WKT_file(f)

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
        return domain_from_basin(geometry["uuid"], name=name)

    raise Exception(
        "The geometry type you have chosen should have been implemented "
        "but, because of a bug of the code, it is not recognized"
    )


def domain_from_string(domain_description: str) -> ConcreteDomain:
    if domain_description.startswith("basin:"):
        basin_uuid = domain_description[len("basin:") :]
        try:
            return domain_from_basin(basin_uuid=basin_uuid, name=basin_uuid)
        except InvalidBasinUUIDError as e:
            raise argparse.ArgumentTypeError(str(e)) from e

    yaml_path = _read_path(domain_description)
    if not yaml_path.is_file():
        raise argparse.ArgumentTypeError(
            f"The provided domain description {domain_description} is not a "
            "valid file path nor a valid basin identifier (it should start "
            "with 'basin:')"
        )

    return read_domain_from_yaml(yaml_path)
