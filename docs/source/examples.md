# Examples

The following section presents three examples demonstrating the complete Medunda workflow, from data download and processing to visualization.

## Extraction of the bottom dissolved oxygen in the Mediterranean Sea

**Downloader command**:
```bash
poetry run medunda downloader create --start-date 1994-01-01 --end-date 2024-12-31 --variables o2  --domain domains/MediterraneanSea.yaml --frequency monthly --provider cmems_mediterranean --split-by year --output-dir "./medunda_dataset/medsea"
```

**Reducer command**:
```bash
poetry run medunda reducer --input-dataset "./medunda_dataset/medsea" --variable o2 --output-file "./medunda_dataset/medsea/bottom_oxygen.nc" --format netcdf  extract_bottom
```

**Plotter command**:

To generate *timeseries*:
```bash
poetry run medunda plotter --input-file "./medunda_dataset/medsea/bottom_oxygen.nc" --variable o2 --show-plot plotting_timeseries
```

To generate *maps*:
```bash
poetry run medunda plotter --input-file "./medunda_dataset/medsea/bottom_oxygen.nc" --variable o2 --show-plot plotting_maps --time 2020-01-01
```

## A scope  of primary production within GSA 9

**Downloader command**:
```bash
poetry run medunda downloader create --start-date 1999-01-01 --end-date 2022-12-31 --variables nppv chl --domain domains/GSA9.yaml --frequency monthly --provider cmems_mediterranean --split-by year --output-dir "./medunda_dataset/gsa9"
```

**Reducer command**:
```bash
poetry run medunda reducer --input-dataset "./medunda_dataset/gsa9" --variable chl --output-file "./medunda_dataset/gsa9/intg_chl.nc" --format netcdf  integrate_between_layers --depth-min 0 --depth-max 200
```

**Plotter command**:

To generate *timeseries*:
```bash
poetry run medunda plotter --input-file "./medunda_dataset/gsa9/intg_chl.nc" --variable chl --show-plot plotting_timeseries
```

To generate *maps*:
```bash
poetry run medunda plotter --input-file "./medunda_dataset/gsa9/intg_chl.nc" --variable chl --show-plot plotting_maps --time 2000-01-01
```

## Global ocean climatology

**Downloader command**:
```bash
poetry run medunda downloader create --start-date 1994-01-01 --end-date 2025-06-30 --variables thetao --domain domains/GlobalOcean.yaml --frequency monthly --provider cmems_global --split-by year --output-dir "./medunda_dataset/global"
```

**Reducer command**:
```bash
poetry run medunda reducer --input-dataset "./medunda_dataset/global" --variable thetao --output-file "./medunda_dataset/global/temperature_climatology.nc" --format netcdf  compute_climatology
 ```

**Plotter command**:

To generate *timeseries*:
```bash
poetry run medunda plotter --input-file "./medunda_dataset/global/temperature_climatology.nc" --variable thetao --show-plot plotting_timeseries
```
