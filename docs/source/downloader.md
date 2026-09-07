(downloaderdoc)=
# Downloader

```{eval-rst}
.. automodule:: medunda.downloader
   :no-members:
```

To run the downloader, provide the following required and optional arguments:

1. **Required — variables**: one or more variables can be downloaded that can be found when running this command:
```bash
   medunda show variables
```
   this prints all the variables supported by Medunda from all providers.

2. **Required — start-date and end-date**: written in this format: YYYY-MM-DD

3. **Optional — frequency**: the available frequencies are daily, weekly, monthly and yearly. The frequency is dependent from the dataset. Not all frequencies are supported by all datasets. The default is monthly.

4. **Required — domain**: choose a predefined domain such as the Mediterranean Sea or Adriatic Sea, or provide a YAML file describing a rectangular, shapefile, WKT, or BitSea basin domain. See **{ref}`domaindoc`** for the supported domain types.

5. **Optional — provider**: providers are divided into two main groups. CMEMS providers or Local Providers. (to be explained in a following page: cmems med reanalysis physical and biogeochemical variables or global reanalysis) The default is `cmems_mediterranean`. Local providers require a provider config, which is
```bash
   medunda show providers
```
This prints the providers Medunda reaches out to to download data.

6. **Optional — provider-config**: only needed by providers that require a configuration file. The default is no configuration file.

7. **Optional — split-by**: this offers to users the possibility to split the dataset files according to temporal resolution: either yearly files or monthly files or the entirety of the dataset in one single file. The default option is split-by year.

8. **Required — output-dir**: the directory where to save the downloaded dataset. this should be a one use only directory, meaning the user cannot create two geo-data collection in the same directory.


The full command to start a new download:

```bash
medunda downloader create \
--variables <variable_names> \
--start-date <YYYY-MM-DD> \
--end-date <YYYY-MM-DD> \
--frequency <daily|weekly|monthly|yearly> \
--domain <domain_file> \
--provider <data_source> \
--provider-config <configuration of the provider> \
--split-by <whole|year|month> \
--output-dir <output_path>
```
*Example*:
```bash
medunda downloader create --variables thetao o2 --start-date 1999-01-01 --end-date 2023-12-31 --frequency monthly --domain domains/GSA9.yaml --split-by year --output-dir ./data/
```

In case of interruption of the downloading process, the tool provides a way to resume it from the point where it stopped. This is more effective if the dataset is downloaded with the split-by year.
This is the command: only the directory to the dataset is needed in this case:

```bash
medunda downloader resume --dataset-dir <path_of_the_dataset>
```

By the end of the download, Medunda creates a GeoDataCollection that summarizes the dataset downloaded.
Ici a détailler plus ce que c'est et ce qu'on peut y trouver + capture pour exemple.
