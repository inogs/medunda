from medunda.tools.lazy_imports import copernicusmarine
from medunda.tools.lazy_imports import dask_array
from medunda.tools.lazy_imports import geopandas
from medunda.tools.lazy_imports import numpy
from medunda.tools.lazy_imports import pandas
from medunda.tools.lazy_imports import pyplot
from medunda.tools.lazy_imports import xarray

xr = xarray
da = dask_array
gpd = geopandas
np = numpy
pd = pandas
plt = pyplot

__all__ = [
    "copernicusmarine",
    "da",
    "dask_array",
    "geopandas",
    "gpd",
    "np",
    "numpy",
    "pandas",
    "pd",
    "plt",
    "pyplot",
    "xarray",
    "xr",
]
