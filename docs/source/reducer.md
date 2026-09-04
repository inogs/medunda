(reducerdoc)=
# Reducer

```{eval-rst}
.. automodule:: medunda.reducer
   :no-members:
```

When running the reducer:

```bash
python -m medunda reducer\
--input-dataset <path_to_dataset>\
--variable <variable>\
--output-file <output_path>\
<action> [action_options]
```

you can choose a subset of variables from the dataset.
Each action defines its own command-line options. See **{ref}actionsdoc** for the available actions and their specific arguments.
Each action return a dataset with the processed data.

## Output formats

### NetCDF

NetCDF is the default output format.

The processed `xarray.Dataset` is written directly to the specified NetCDF file.

---

### CSV

CSV output is intended for results containing a single variable.

For CSV output, the result must contain exactly one data variable. The resulting data are converted to a pandas DataFrame and written to the output file.

---

### GeoTIFF

GeoTIFF output is intended for spatial results.

If the result contains a `time` dimension, the reducer first calculates the mean over time.

The spatial dimensions are expected to be:

* `longitude`
* `latitude`

The output is written using the WGS84 coordinate reference system (`EPSG:4326`).

---

### ASCII

ASCII output is intended for two-dimensional spatial grids.

The result must contain a single data variable and the `latitude` and `longitude` dimensions.

If a `time` dimension is present, the reducer calculates the mean over time before creating the ASCII grid.

Missing values are written using `-9999` as the `NODATA` value.
