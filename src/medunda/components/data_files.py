from abc import ABC
from datetime import datetime
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import Discriminator
from pydantic import Tag

from medunda.tools.typing import VarName


class DataSource(BaseModel, ABC):
    """
    Abstract base class for every source of data of a variable.

    Each concrete subclass must define a `data_type` field with a unique
    `Literal` value; this value is used to discriminate among the different
    subclasses when a `DataSource` is deserialized (see `AnyDataSource`).

    Attributes:
        data_type: A string that identifies the concrete type of the source.
        start_date: The first date covered by the source.
        end_date: The last date covered by the source.
        variable: The name of the variable contained in the source.
    """

    data_type: str
    start_date: datetime
    end_date: datetime
    variable: VarName

    def __new__(cls, *args, **kwargs):
        if cls is DataSource:
            raise TypeError(
                "DataSource is an abstract class and can not be instantiated "
                "directly; use one of its subclasses instead"
            )
        return super().__new__(cls)


class NetcdfArchive(DataSource):
    """
    A `DataSource` that stores its data inside a NetCDF file.

    Attributes:
        path: The path of the NetCDF file; if relative, it is interpreted as
            relative to the main path of the collection that contains it.
    """

    data_type: Literal["netcdf_archive"] = "netcdf_archive"
    path: Path


def _get_data_type(value: Any) -> str | None:
    # Files written before the introduction of the `data_type` field do not
    # contain it; all of them describe NetCDF archives.
    if isinstance(value, dict):
        return value.get("data_type", "netcdf_archive")
    return getattr(value, "data_type", None)


# Use this type (instead of `DataSource`) for the fields of pydantic models
# that must be deserialized into the right `DataSource` subclass. Every new
# subclass of `DataSource` must be added here, together with its tag, e.g.
#   Annotated[NetcdfArchive, Tag("netcdf_archive")]
#   | Annotated[OtherSource, Tag("other_source")],
AnyDataSource = Annotated[
    Annotated[NetcdfArchive, Tag("netcdf_archive")],
    Discriminator(_get_data_type),
]
