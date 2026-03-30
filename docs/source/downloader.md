(downloaderdoc)=
# Downloader

```{eval-rst}
.. automodule:: medunda.downloader
   :no-members:
```


To start a new download:

```bash
python -m medunda downloader create --variable <variable_name> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD> --frequency <daily/monthly> --domain <domain_file> --provider <data_source> --provider-config <configuration of the provider> --split-by <whole/year/month> --output-dir <output_path>
```

To resume an interrupted download:
```bash
python -m medunda downloader resume --dataset-path <path_of_the_dataset>
```
*Example*:
```bash
python -m medunda.downloader --variable thetao --start-date 1999-01-01 --end-date 2023-12-31 --frequency monthly --domain gsa9 --split-by year --output-dir ./data/
```
