(plotterdoc)=
# Plotter

```{eval-rst}
.. automodule:: medunda.plotter
   :no-members:
```
The plotter is a visualization tool used to generate plots from data produced by the Medunda downloader or reducer.

It supports the following input formats:

* NetCDF (`.nc`)
* CSV (`.csv`)
* GeoTIFF (`.tif`)

Two plotting modes are available:

* `plotting_timeseries`: generates a time series plot over a selected time range.
* `plotting_maps`: generates a spatial map for a selected time.

The generated plots can either be saved to an output directory or displayed in an interactive window using `--show-plot`.

```{eval-rst}
.. automodule:: medunda.plotter
   :no-members:
```

The plotter can be executed from the command line using:

```bash
poetry run medunda plotter \
    --input-file <path_to_file> \
    --variable <variable_name> \
    --output-dir <output_path> \
    <mode> [mode_options]
```

The main arguments are:

* `--input-file`: path to the input file.
* `--variable`: name of the variable to plot.
* `--output-dir`: directory where the generated plot is saved.
* `--show-plot`: display the plot in an interactive window instead of saving it to a file.
* `<mode>`: plotting mode to execute.
* `[mode_options]`: additional options required by the selected mode.

---

## plotting_timeseries

The `plotting_timeseries` mode generates a time series plot for a selected variable over a specified time range.

```bash
python -m medunda plotter \
    --input-file <path_to_file> \
    --variable <variable_name> \
    --output-dir <output_path> \
    plotting_timeseries \
    --start-time <YYYY-MM-DD> \
    --end-time <YYYY-MM-DD>
```

The `--start-time` and `--end-time` options define the period to be plotted.

Time series plots are supported for:

* NetCDF files
* CSV files

Time series plotting is not supported for GeoTIFF files.

---

## plotting_maps

The `plotting_maps` mode generates a spatial map of the selected variable for a specified time.

```bash
python -m medunda plotter \
    --input-file <path_to_file> \
    --variable <variable_name> \
    --output-dir <output_path> \
    plotting_maps \
    --time <YYYY-MM-DD>
```

The `--time` option specifies the date to be plotted.

For variables with a vertical dimension, the aggregation dimension and aggregation method can also be specified:

```bash
python -m medunda plotter \
    --input-file <path_to_file> \
    --variable <variable_name> \
    --output-dir <output_path> \
    plotting_maps \
    --time <YYYY-MM-DD> \
    --aggregation-dimension <dimension> \
    --aggregation-method <method>
```

Spatial map plotting is supported for:

* NetCDF files
* GeoTIFF files

Plotting maps from CSV files is not supported.

---

## Input formats

### NetCDF

NetCDF files (`.nc`) support both plotting modes:

* `plotting_timeseries`
* `plotting_maps`

The selected variable must be present in the dataset. For time series plots, the variable must also contain a `time` dimension.

---

### CSV

CSV files (`.csv`) can be used to generate time series plots.

The CSV file must contain the selected variable. If a `time` column is present, it is automatically converted to a datetime index.

Spatial map plotting from CSV files is not supported.

---

### GeoTIFF

GeoTIFF files (`.tif`) can be used to generate spatial maps.

Time series plotting from GeoTIFF files is not supported.

---

## Displaying plots

By default, plots are saved to the directory specified with `--output-dir`.

Alternatively, the `--show-plot` option can be used to display the plot in an interactive window:

```bash
python -m medunda plotter \
    --input-file <path_to_file> \
    --variable <variable_name> \
    --show-plot \
    plotting_timeseries \
    --start-time <YYYY-MM-DD> \
    --end-time <YYYY-MM-DD>
```
---
