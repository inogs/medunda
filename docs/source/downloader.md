(downloaderdoc)=
# Downloader

```{eval-rst}
.. automodule:: medunda.downloader
   :no-members:
```

To run the downloader the following arguments are required:

1. variables: one or more variables can be downloaded that can be found when running this command:
```bash
   medunda show variables
```
   this prints all the variables supported by Medunda from all providers.

2. start-date and end-date written in this format: YYYY-MM-DD

3. frequency: the available frequencies are daily, weekly, monthly and yearly. The frequency is dependent from the dataset. Not all frequencies are supported by all datasets

4. domain: Medunda offers a set of predefined domains such as the Mediterranean Sea or the Adritic Sea that the user can choose from. However, the user can create his own domain, like the following: through three methods: a yaml file with the exact coordinates following a rectangular domain. a shapefile, Medunda supports all GSAs across the Mediterranean Sea, always on a yaml file, a wkt (csv) file from an online map or from bit.sea either a simple or complex domain. (need a proper page/link to more explain this and give examples)

5. provider: providers are divided into two main groups. CMEMS providers or Local Providers. (to be explained in a following page: cmems med reanalysis physical and biogeochemical variables or global reanalysis) while local providers require a provider config which is
```bash
   medunda show
```
This prints the providers Medunda reaches out to to download data.

6. provider_config: to be chosen from the local datasets pre defined by Medunda already in a specific folder called providers_config

7. split by: this offers to users the possibility to split the dataset files according to temporal resolution: either yearly files or monthly files or the entirety of the dataset in one single file. the default option is split-by year.

8. output-dir: the directory where to save the downloaded dataset. this should be a one use only directory, meaning the user cannot create two geo-data collection in the same directory.


The full command to start a new download:

```bash
medunda downloader create\
--variables <variable_names>\
--start-date <YYYY-MM-DD>\
--end-date <YYYY-MM-DD>\
--frequency <daily|weekly|monthly|yearly>\
--domain <domain_file>\
--provider <data_source>\
--provider-config <configuration of the provider>\
--split-by <whole|year|month>\
--output-dir <output_path>
```
*Example*:
```bash
medunda downloader create --variables thetao o2 --start-date 1999-01-01 --end-date 2023-12-31 --frequency monthly --domain gsa9 --split-by year --output-dir ./data/
```

In case of interruption of the downloading process, the tool provides a way to resume it from the point where it stopped. This is more effective if the dataset is downloaded with the split-by year.
This is the command: only the directory to the dataset is needed in this case:

```bash
medunda downloader resume --dataset-dir <path_of_the_dataset>
```

By the end of the download, Medunda creates a GeoDataCollection that summarizes the dataset downloaded.
Ici a détailler plus ce que c'est et ce qu'on peut y trouver + capture pour exemple.
