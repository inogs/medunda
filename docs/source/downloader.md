(downloaderdoc)=
# Downloader

```{eval-rst}
.. automodule:: medunda.downloader
   :no-members:
```


To start a new download:

```bash
medunda downloader create --variables <variable_name> --start-date <YYYY-MM-DD> --end-date <YYYY-MM-DD> --frequency <daily|weekly|monthly|yearly> --domain <domain_file> --provider <data_source> --provider-config <configuration of the provider> --split-by <whole|year|month> --output-dir <output_path>
```

To resume an interrupted download:
```bash
medunda downloader resume --dataset-dir <path_of_the_dataset>
```

*Example*:
```bash
medunda downloader create --variables thetao o2 --start-date 1999-01-01 --end-date 2023-12-31 --frequency monthly --domain gsa9 --split-by year --output-dir ./data/
```
