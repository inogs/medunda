(reducerdoc)=
# Reducer

```{eval-rst}
.. automodule:: medunda.reducer
   :no-members:
```

The reducer can be executed from the command line using:

```bash
python -m medunda reducer \
    --input-dataset <path_to_dataset> \
    --variables <variable_names> \
    --output-file <output_path> \
    --format <netcdf|csv|geotiff|ascii> \
    <action> [action_options]
```

The main arguments are:

* `--input-dataset`: path of the Medunda dataset to be processed, i.e. the directory that was passed as `--output-dir` to the **{ref}`downloaderdoc`**.
* `--variables`: one or more variables to be processed. This argument is optional: if it is not specified, all the variables of the dataset are processed.
* `--output-file`: path of the output file.
* `--format`: format of the output file. This argument is optional; the default is `netcdf`. The available formats are described below.
* `<action>`: the operation to execute.
* `[action_options]`: additional options required by the selected action.

Before executing the action, the reducer masks (i.e. sets to `NaN`) all the values that lie outside the domain of the dataset (see **{ref}`domaindoc`**).

Each action defines its own command-line options. See **{ref}`actionsdoc`** for the available actions and their specific arguments.
Each action returns a dataset with the processed data, which can then be visualized using the **{ref}`plotterdoc`**.

## Output formats

### NetCDF

[NetCDF](https://www.unidata.ucar.edu/software/netcdf/) is the default output format.

The processed [`xarray.Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) is written directly to the specified NetCDF file.

---

### CSV

CSV output is intended for results containing a single variable.

For CSV output, the result must contain exactly one data variable; use `--variables` to select it. The resulting data are converted to a [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html), with one column for each coordinate and one for the variable, and written to the output file.

---

### GeoTIFF

[GeoTIFF](https://www.ogc.org/standards/geotiff/) output is intended for spatial results.

If the result contains a `time` dimension, the reducer first calculates the mean over time.

The spatial dimensions are expected to be:

* `longitude`
* `latitude`

The output is written using the WGS84 coordinate reference system ([`EPSG:4326`](https://epsg.io/4326)).

---

### ASCII

ASCII output produces a grid in the [Esri ASCII raster format](https://desktop.arcgis.com/en/arcmap/latest/manage-data/raster-and-images/esri-ascii-raster-format.htm) and is intended for two-dimensional spatial grids.

The result must contain a single data variable (use `--variables` to select it) and the `latitude` and `longitude` dimensions.

If a `time` dimension is present, the reducer calculates the mean over time before creating the ASCII grid. After this step, the data must be two-dimensional: variables that also have a `depth` dimension must first be reduced with an action that removes it (for example, `extract_surface` or `compute_depth_average`).

Missing values are written using `-9999` as the `NODATA` value.
