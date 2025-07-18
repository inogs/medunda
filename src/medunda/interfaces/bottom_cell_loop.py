from datetime import timedelta
from typing import Any
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
from bitsea.commons.mask import Mask
from bitsea.commons.geodistances import compute_geodesic_distance

from medunda.dataset import Dataset
from medunda.dataset import read_dataset


class BottomCellLoop:
    def __init__(self,
                 dataset: Dataset,
                 time_range: timedelta,
                 output_dtypes: dict[str, np.typing.DTypeLike] | None = None,
                 lat_column: str = "latitude",
                 lon_column: str = "longitude",
                 time_column: str = "time",
                 ):
        self._dataset = dataset
        self._time_range = time_range
        self._output_dtypes = output_dtypes
        self._lat_column = lat_column
        self._lon_column = lon_column
        self._time_column = time_column

    def map(self, func: Callable[[xr.Dataset], dict[str, Any]], point_table: pd.DataFrame) -> pd.DataFrame:
        mask = Mask.from_xarray(self._dataset.get_mask())
        bottom_indices: np.ndarray = mask.bathymetry_in_cells() - 1

        def get_model_indices(row):
            lat = row[self._lat_column]
            lon = row[self._lon_column]
            # Convert the point's latitude and longitude to the nearest model point indices
            model_lon_index, model_lat_index = mask.convert_lon_lat_wetpoint_indices(
                lat=lat, lon=lon, max_radius=None,
            )
            return pd.Series([model_lon_index, model_lat_index])

        point_table[["model_lon_index", "model_lat_index"]] = point_table.apply(
            get_model_indices, axis=1
        )

        point_table["model_lat"] = mask.lat[point_table["model_lat_index"]]
        point_table["model_lon"] = mask.lon[point_table["model_lon_index"]]

        point_table["distance_from_model"] = compute_geodesic_distance(
            lat1=point_table[self._lat_column],
            lon1=point_table[self._lon_column],
            lat2=point_table["model_lat"],
            lon2=point_table["model_lon"],
        )

        data = self._dataset.get_data()

        results = []
        for point in point_table.itertuples():
            point_time = getattr(point, self._time_column)
            lat_index: int = getattr(point, "model_lat_index")
            lon_index: int = getattr(point, "model_lon_index")
            time_slice = slice(
                point_time - self._time_range,
                point_time + timedelta(milliseconds=1)
            )
            point_data = data.isel(
                longitude=point.model_lon_index,
                latitude=point.model_lat_index,
                depth=bottom_indices[lat_index, lon_index],
            ).sel(time=time_slice)
            point_data = point_data.rename(
                {
                    "latitude": "model_latitude",
                    "longitude": "model_longitude",
                    "depth": "model_depth",
                    "time": "model_time",
                }
            )
            point_data = point_data.assign_coords(
                latitude=getattr(point, self._lat_column),
                longitude=getattr(point, self._lon_column),
                time=getattr(point, self._time_column),
                distance=point.distance_from_model,
            )
            point_data = point_data.compute()
            output_raw = func(point_data)

            if self._output_dtypes is not None:
                output = {}
                for k in self._output_dtypes:
                    output[k] = output_raw[k]
            else:
                output = output_raw

            output[self._lat_column] = getattr(point, self._lat_column)
            output[self._lon_column] = getattr(point, self._lon_column)
            output[self._time_column] = point.time
            results.append(output)

        output_dataframe = pd.DataFrame(results)
        if self._output_dtypes is not None:
            for k, v in self._output_dtypes.items():
                output_dataframe[k] = output_dataframe[k].astype(v) # type: ignore

        # Reorder the columns so that they are coherent with the original format
        reordered_columns = [
            t for t in point_table.columns if t in output_dataframe.columns
        ]
        reordered_columns.extend(
            [t for t in output_dataframe.columns if t not in point_table.columns]
        )
        output_dataframe = output_dataframe[reordered_columns]

        return output_dataframe

