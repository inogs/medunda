import argparse
import logging
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import cast

import medunda.tools.lazy_imports.cmocean as lazy_cmocean
import medunda.tools.lazy_imports.colormap as clp
import medunda.tools.lazy_imports.rioxarray as rioxarray
from medunda.components.variables import Variable
from medunda.components.variables import VariableDataset
from medunda.plots import maps
from medunda.plots import timeseries
from medunda.tools.lazy_imports import pd
from medunda.tools.lazy_imports import xr
from medunda.tools.logging_utils import configure_logger

if TYPE_CHECKING:
    from matplotlib.colors import Colormap


LOGGER = logging.getLogger(__name__)

VAR_METADATA = MappingProxyType(
    {
        "o2": MappingProxyType(
            {"label": "Oxygen", "unit": "µmol/m³", "cmap_name": "cmo:deep"}
        ),
        "chl": MappingProxyType(
            {
                "label": "Chlorophyll-a",
                "unit": "mg/m³",
                "cmap_name": "cmo:algae",
            }
        ),
        "nppv": MappingProxyType(
            {
                "label": "Net Primary Production",
                "unit": "mg C/m²/day",
                "cmap_name": "cmo:matter",
            }
        ),
        "thetao": MappingProxyType(
            {"label": "Temperature", "unit": "°C", "cmap_name": "coolwarm"}
        ),
        "so": MappingProxyType(
            {"label": "Salinity", "unit": "PSU", "cmap_name": "viridis"}
        ),
    }
)

DEFAULT_VAR = MappingProxyType(
    {
        "unit": "",
        "cmap_name": "viridis",
    }
)


PLOTS = [
    timeseries,
    maps,
]


def configure_parser(
    parser: argparse.ArgumentParser | None = None,
) -> argparse.ArgumentParser:
    """
    parse command line arguments:
    --input-file: path of the input file
    --variable: name of the variable to plot
    --output-dir: directory to save the download file
    """
    if parser is None:
        parser = argparse.ArgumentParser(
            description="plots timeseries and maps"
        )

    parser.add_argument(
        "--input-file", type=Path, required=True, help="Path of the input file"
    )
    parser.add_argument(
        "--variable",
        type=str,
        choices=VariableDataset.all_variables().get_variable_names(),
        required=False,
        help="Name of the variable to plot",
    )
    # parser.add_argument(
    #     "--output-dir",
    #     type=Path,
    #     default=Path("."),
    #     help="Directory where the downloaded files are saved",
    # )

    subparsers = parser.add_subparsers(
        title="mode",
        required=True,
        dest="mode",
        help="Sets which plot must be executed on the input file",
    )

    for mode in PLOTS:
        LOGGER.debug("Letting module %s configure its parser", mode.__name__)
        mode.configure_parser(subparsers)

    return parser


def parse_colormap_description(cmap_desc: str) -> "Colormap":
    if cmap_desc.startswith("cmo:"):
        return lazy_cmocean.get_cmocean_map(cmap_desc[4:])
    else:
        return clp.Colormap(cmap_desc)


def check_variable(ds, var):
    if isinstance(ds, xr.Dataset):
        if var not in ds:
            raise ValueError(f"Variable '{var}' not found in dataset")
        data = ds[var]

    elif isinstance(ds, xr.DataArray):
        data = ds

    else:
        raise ValueError("Unsupported dataset type")

    if var in VAR_METADATA:
        var_metadata: dict[str, "str | Colormap"] = dict(VAR_METADATA[var])
    else:
        var_metadata: dict[str, "str | Colormap"] = dict(DEFAULT_VAR)
        var_metadata["label"] = var

    cmap_name = cast(str, var_metadata["cmap_name"])
    del var_metadata["cmap_name"]

    var_metadata["cmap"] = parse_colormap_description(cmap_name)

    return data, var_metadata


def plotter(filepath: Path, variable: str, mode: str, args):
    """The main function of the plotter module, which is responsible for plotting the data contained in the input file.
    Depending on the mode specified by the user, this function calls the appropriate plotting function to plot the data.
    The plotting functions are defined in the `plots` module and are responsible for plotting timeseries and maps.
    Extracts and plots surface, bottom, and average layers of the given variable."""

    if not filepath.exists():
        raise FileNotFoundError(f"The file '{filepath}' does not exist.")

    if filepath.suffix == ".nc":
        with xr.open_dataset(filepath) as ds:
            data_var, metadata = check_variable(ds, variable)

        if mode == "plotting_timeseries":
            if "time" not in ds[variable].dims:
                raise ValueError(
                    f"Variable '{variable} does not have the time dimension"
                )
            timeseries.plotting_timeseries(
                data=data_var,
                metadata=metadata,
                start_date=args.start_date,
                end_date=args.end_date,
            )

        elif mode == "plotting_maps":
            maps.plotting_maps(
                data=data_var,
                metadata=metadata,
                time=args.time,
                aggregation_dimension=args.aggregation_dimension,
                aggregation_method=args.aggregation_method,
            )
        else:
            raise ValueError("Invalid mode")

    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath)

        if "time" in df.columns:
            # Ensure that the time column is in datetime format
            # and set the time column as the index
            df["time"] = pd.to_datetime(df["time"])
            df.set_index("time", inplace=True)
            df.sort_index(inplace=True)
        else:
            df.set_index(
                [col for col in df.columns if col != variable], inplace=True
            )

        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' not found in CSV")
        else:
            metadata: dict[str, "str | Colormap"] = dict(DEFAULT_VAR)
            try:
                var_object = Variable.get_by_name(variable)
                metadata["label"] = var_object.get_label()
            except ValueError:
                metadata["label"] = variable

            cmap_name = cast(str, metadata["cmap_name"])
            metadata["cmap"] = parse_colormap_description(cmap_name)
            del metadata["cmap_name"]

        ds = df.to_xarray()

        data_var = ds[variable]
        data_var, _ = check_variable(ds, variable)

        if mode == "plotting_timeseries":
            if "time" not in ds[variable].dims:
                raise ValueError(
                    f"Variable '{variable} does not have the time dimension"
                )
            else:
                timeseries.plotting_timeseries(
                    data=data_var,
                    metadata=metadata,
                    start_date=args.start_date,
                    end_date=args.end_date,
                )
        elif mode == "plotting_maps":
            raise ValueError("Plotting maps from CSV files is not supported")
        else:
            raise ValueError("Invalid mode")

    elif filepath.suffix == ".tif":
        with rioxarray.open_rasterio(filepath) as ds:
            print(ds)

            data_var, metadata = check_variable(ds, variable)

            if mode == "plotting_timeseries":
                raise ValueError(
                    "Plotting timeseries from GeoTIFF files is not supported"
                )

            elif mode == "plotting_maps":
                maps.plotting_maps(
                    data=data_var,
                    metadata=metadata,
                    time=args.time,
                    aggregation_dimension=args.aggregation_dimension,
                    aggregation_method=args.aggregation_method,
                )
            else:
                raise ValueError("Invalid mode")

    else:
        raise ValueError(
            f"Unsupported file format '{filepath.suffix}'. Supported formats are: .nc, .csv, .tif"
        )

    return 0


def main():
    args = configure_parser().parse_args()
    configure_logger(LOGGER)

    # output_dir = args.output_dir
    variable = args.variable
    mode = args.mode
    data_file = args.input_file

    LOGGER.info(f"Selected file: {data_file.name}")

    plotter(filepath=data_file, variable=variable, mode=mode, args=args)

    LOGGER.info(
        f"Plotting completed for variable '{variable}' in mode '{mode}'"
    )


if __name__ == "__main__":
    main()
