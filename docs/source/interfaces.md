(interfacesdoc)=
# Python Interface

```{eval-rst}
.. automodule:: medunda.interfaces.bottom_cell_map
   :no-members:
```

## Bottom Cell Map

`BottomCellMap` maps observation points to the corresponding bottom cell of an oceanographic model grid and extracts the associated temporal data.

The input points are provided as a Pandas DataFrame containing latitude, longitude, and time information. For each point, the corresponding model grid cell is identified based on the nearest valid surface cell and the model bathymetry. Data are then extracted from the deepest valid cell at that location.

The mapping can be performed immediately or using Dask for delayed and parallel computation.

```{eval-rst}
.. autoclass:: medunda.interfaces.bottom_cell_map.BottomCellMap
   :members:
   :show-inheritance:
```

### Example

A `BottomCellMap` object can be initialized with a `GeoDataCollection` and a time range:

```python
from datetime import timedelta

from medunda.interfaces.bottom_cell_map import BottomCellMap

bottom_mapper = BottomCellMap(
    dataset=dataset,
    time_range=timedelta(days=7),
)
```

The mapping is then performed by providing a function and a table of points:

```python
result = bottom_mapper.map(
    func=my_function,
    point_table=points,
)
```

The user-defined function receives an `xarray.Dataset` containing the data extracted for each point and must return a dictionary. The dictionary keys are used as column names in the output.

For example:

```python
def my_function(dataset):
    return {
        "temperature": dataset["thetao"].mean().item(),
    }
```

The resulting DataFrame contains the original point information together with the columns returned by the function.

---

## Output data

For each point, the dataset provided to the user-defined function contains the temporal data extracted from the corresponding bottom cell.

In addition to the model variables, the dataset contains information describing the relationship between the observation point and the model grid:

* `latitude`: latitude of the original point.
* `longitude`: longitude of the original point.
* `time`: time of the original point.
* `model_latitude`: latitude of the corresponding model grid cell.
* `model_longitude`: longitude of the corresponding model grid cell.
* `model_depth`: depth of the corresponding bottom cell.
* `distance`: geodesic distance between the original point and the corresponding model grid cell, in metres.

The extracted model data use the `model_time` dimension.

---

## Delayed computation

By default, `map()` computes the results immediately and returns a Pandas DataFrame.

For large datasets, delayed computation can be enabled:

```python
result = bottom_mapper.map(
    func=my_function,
    point_table=points,
    delayed=True,
)
```

In this case, the computation is returned as a Dask DataFrame and can be executed later using `.compute()`.

---

## Custom column names

The names of the latitude, longitude, and time columns in the input point table can be customized when creating the `BottomCellMap` object:

```python
bottom_mapper = BottomCellMap(
    dataset=dataset,
    time_range=timedelta(days=7),
    lat_column="lat",
    lon_column="lon",
    time_column="date",
)
```

The corresponding columns must exist in the input point table.
