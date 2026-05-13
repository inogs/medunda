import medunda.tools.lazy_imports.copernicusmarine as copernicusmarine
import medunda.tools.lazy_imports.dask_array as dask_array
import medunda.tools.lazy_imports.geopandas as geopandas
import medunda.tools.lazy_imports.numpy as numpy
import medunda.tools.lazy_imports.pandas as pandas
import medunda.tools.lazy_imports.pyplot as pyplot
import medunda.tools.lazy_imports.xarray as xarray

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
    "xarray",
    "xr",
    "gpd",
    "geopandas",
    "pandas",
    "pd",
    "plt",
    "pyplot",
    "numpy",
    "np",
]
