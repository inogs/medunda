import logging
import shutil
from datetime import timezone
from pathlib import Path
from typing import Mapping

import copernicusmarine

from medunda.components.data_files import DataFile
from medunda.components.frequencies import Frequency
from medunda.domains.domain import Domain
from medunda.tools.file_names import from_file_path_to_time_range
from medunda.tools.typing import VarName
from medunda.providers.provider import Provider


LOGGER = logging.getLogger(__name__)


PRODUCTS={
    #"MEDSEA_MULTIYEAR_PHY_006_004":
    ("thetao",):
        {Frequency.DAILY: "med-cmcc-tem-rean-d",
        Frequency.MONTHLY: "med-cmcc-tem-rean-m"},
    ("vo", "uo"):
        {Frequency.DAILY: "med-cmcc-cur-rean-d",
        Frequency.MONTHLY:"med-cmcc-cur-rean-m"},
    ("so",):
        {Frequency.DAILY: "med-cmcc-sal-rean-d",
        Frequency.MONTHLY: "med-cmcc-sal-rean-m"},

    #"MEDSEA_MULTIYEAR_BGC_006_008":
    ("ph",):
         {Frequency.DAILY: "med-ogs-car-rean-d",
         Frequency.MONTHLY: "med-ogs-car-rean-m"},
    ("no3","po4","si"):
        {Frequency.DAILY: "med-ogs-nut-rean-d",
         Frequency.MONTHLY: "med-ogs-nut-rean-m"},
    ("chl",):
        {Frequency.DAILY: "med-ogs-pft-rean-d",
         Frequency.MONTHLY: "med-ogs-pft-rean-m"},
    ("o2",):
        {Frequency.DAILY: "med-ogs-bio-rean-d",
         Frequency.MONTHLY: "med-ogs-bio-rean-m"},
    ("nppv",):
         {Frequency.DAILY: "med-ogs-bio-rean-d",
         Frequency.MONTHLY: "med-ogs-bio-rean-m"},
}

VARIABLES = []
for _var_list in PRODUCTS.keys():
    VARIABLES.extend(_var_list)
VARIABLES.sort()


def search_for_product(var_name: VarName, frequency:Frequency) -> str:
    """ Given the name of a variable and a frequency, return the name of the CMEMS product that
    contains such variable with the specified frequency."""

    selected_product = None
    for vars_tuple, prod_dict in PRODUCTS.items():
        if var_name in vars_tuple:
            selected_product = prod_dict[frequency]
            break

    if selected_product is None:
        raise ValueError (f"Variable '{var_name}' is not available in the dictionary")

    LOGGER.debug(f"var_name={var_name}, selected_product={selected_product}")

    return selected_product


class CMEMSProvider(Provider):
    @classmethod
    def get_name(cls) -> str:
        return "cmems"

    @classmethod
    def create(cls, config_file: Path | None = None) -> "Provider":
        if config_file is not None:
            raise ValueError("CMEMS provider does not support configuration files")
        return cls()

    def download_data(self, domain: Domain, frequency: Frequency, data_files: Mapping[VarName, tuple[DataFile, ...]]) -> None:
        """
        Downloads the data for the dataset.

        This method downloads all the missing data files for the dataset.
        If a file already exists, it will not be downloaded again.
        """
        for var_name, files in data_files.items():
            LOGGER.debug(
                f'Downloading data for variable "%s": %s files will be '
                f'downloaded',
                var_name,
                len(files)
            )

            product_id = search_for_product(
                var_name=var_name,
                frequency=frequency
            )
            LOGGER.debug(
                "The product associated with variable %s is: %s",
                var_name,
                product_id
            )

            for file in files:
                if file.path.exists():
                    LOGGER.debug(
                        'File "%s" already exists. Skipping download.', file.path
                    )
                    continue

                LOGGER.info('Downloading file "%s"', file.path)

                # Ensure the parent directory exists
                file_path = file.path.absolute()
                file_path.parent.mkdir(parents=True, exist_ok=True)

                output_filename = file_path.name
                temp_filename = "tmp." + output_filename
                temp_file_path = file_path.parent / temp_filename

                if temp_file_path.is_file():
                    LOGGER.debug(
                        '"%s" already exists. I will delete it!',
                        temp_file_path
                    )
                    temp_file_path.unlink()

                # Copernicusmarine API interprets datetimes as UTC,
                # so we need to ensure that the start and end datetimes
                # are timezone-aware and set to UTC.
                start = file.start_date.replace(tzinfo=timezone.utc)
                end = file.end_date.replace(tzinfo=timezone.utc)
                copernicusmarine.subset(
                    dataset_id=product_id,
                    variables=[var_name],
                    start_datetime=start,
                    end_datetime=end,
                    output_filename=str(temp_file_path),
                    **domain.model_dump(exclude={"name"}, exclude_none=True)
                )

                LOGGER.debug(
                    'Moving file "%s" to "%s"', temp_file_path, file_path
                )
                shutil.move(temp_file_path, file_path)
