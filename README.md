# MedUnda

## Mediterranean Engine for Downloading, Understanding and Navigating Data and Analysis

A toolkit for downloading and analyzing oceanographic data from CMEMS in the Mediterranean Sea

### What is Medunda for?

Medunda is designed to automate the downloading, processing, and visualization of environmental data for the **Mediterranean Sea**, with a focus on both **physical and biogeochemical variables** provided by the **Copernicus Marine Environment Monitoring Service (CMEMS)**.
It offers structured access to these datasets and helps researchers efficiently prepare data for use in ecosystem models.
The toolkit is user-friendly and flexible, allowing users to customize data downloads and processing based on their needs and choices.

### How do I use Medunda?

Medunda consists of **three main tools**: 

* Downloader: downloads and organize environmental data
* Reducer: preprocess and transform downloaded datasets and extracts data accordingly
* Plotter: visualizes data

**Downloader command**:

python -m medunda.downloader --variable <variable_name> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD> --frequency <daily/monthly> --domain <domain_file> --split-by <whole/year/month> --output-dir <output_path>

*Example*:
python -m medunda.downloader --variable thetao --start-date 1999-01-01 --end-date 2023-12-31 --frequency monthly --domain gsa9 --split-by year --output-dir ./data/

**Reducer command**:

python -m medunda.reducer --input-dataset <path_to_datasetfile.nc> --output-file <output_path> <action> [action_options]

available actions: 

* extract_bottom: extracts the bottom layer 
* extract_surface: extracts the surface layer
* extract_layer: extracts a specific layer at a specific depth 
* averaging_between_layers: compute the vertical average between two specific depths
* depth_average: compute the average on all the vertical levels
* calculate_stats: calculates the values of some specific statistical operations: [mean, median, variance, quartiles, minimum, maximum, all]
* extract_extremes: extracts the maximum and the minimum values of a variable for each year
* extract_layer_extremes: extract the minimum and maximum value of a variable for each layer available in the dataset 

**Plotter command**:

python -m medunda.plotter --input-file <path_to_file.nc> --variable <variable_name> --output-dir <output_path> <mode> [mode_options]

available modes:

* plotting_timeseries: generate a time series plot for the downloaded domain at a specific time range from --start-time <YYYY-MM-DD> to --end-time <YYYY-MM-DD>
* plotting_maps: create a spatial map for a selected time --time <YYYY-MM-DD>

*To see all available options, requird inputs and usage instructions, run:*
*python -m medunda.<tool_name> -h*

#### What does medunda produce? 

Medunda generates different types of outputs depending on which tool and operation are used.

1. The downloader retrieves environmental data and stores it in an organized folder structure based on the variable and the time resolution.  
The downloaded dataset is structured this way:
/thetao/
    ├── daily/
    └── monthly/
        └── thetao_1999-01-01_1999-12-31.nc
        └── thetao_2000-01-01_2000-12-31.nc
        .
        .
        .
        └── thetao_2023-01-01_2023-12-31.nc

A metadata JSON file is automatically created alongside the downloaded data. It contains a summary of information about the spatial domain, time range covered, time resolution, source of the data and the local file paths to the corresponding data. 

2. The Reducer performs transformations on the downloaded datasets. 

3. The Plotter creates visual representations of the data for quick inspection of the data.