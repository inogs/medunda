import logging
import shutil
from abc import ABC
from abc import abstractmethod
from datetime import timezone
from pathlib import Path
from typing import Mapping

import copernicusmarine

from medunda.components.data_files import DataFile
from medunda.components.frequencies import Frequency
from medunda.components.variables import VariableDataset
from medunda.domains.domain import Domain
from medunda.providers.provider import Provider
from medunda.tools.typing import VarName

LOGGER = logging.getLogger(__name__)


MED_PRODUCTS = {
    # "MEDSEA_MULTIYEAR_PHY_006_004":
    ("thetao",): {
        Frequency.DAILY: "med-cmcc-tem-rean-d",
        Frequency.MONTHLY: "med-cmcc-tem-rean-m",
        Frequency.YEARLY: "cmems_mod_med_phy-tem_my_4.2km_P1Y-m",
    },
    ("vo", "uo"): {
        Frequency.DAILY: "med-cmcc-cur-rean-d",
        Frequency.MONTHLY: "med-cmcc-cur-rean-m",
        Frequency.YEARLY: "cmems_mod_med_phy-cur_my_4.2km_P1Y-m",
    },
    ("so",): {
        Frequency.DAILY: "med-cmcc-sal-rean-d",
        Frequency.MONTHLY: "med-cmcc-sal-rean-m",
        Frequency.YEARLY: "cmems_mod_med_phy-sal_my_4.2km_P1Y-m",
    },
    ("mlotst",): {
        Frequency.DAILY: "med-cmcc-mld-rean-d",
        Frequency.MONTHLY: "med-cmcc-mld-rean-m",
        Frequency.YEARLY: "cmems_mod_med_phy-mld_my_4.2km_P1Y-m",
    },
    # "MEDSEA_MULTIYEAR_BGC_006_008":
    ("ph",): {
        Frequency.DAILY: "med-ogs-car-rean-d",
        Frequency.MONTHLY: "med-ogs-car-rean-m",
        Frequency.YEARLY: "cmems_mod_med_bgc-car_my_4.2km_P1Y-m",
    },
    ("no3", "po4", "si"): {
        Frequency.DAILY: "med-ogs-nut-rean-d",
        Frequency.MONTHLY: "med-ogs-nut-rean-m",
        Frequency.YEARLY: "cmems_mod_med_bgc-nut_my_4.2km_P1Y-m",
    },
    ("chl",): {
        Frequency.DAILY: "med-ogs-pft-rean-d",
        Frequency.MONTHLY: "med-ogs-pft-rean-m",
        Frequency.YEARLY: "cmems_mod_med_bgc-plankton_my_4.2km_P1Y-m",
    },
    ("o2",): {
        Frequency.DAILY: "med-ogs-bio-rean-d",
        Frequency.MONTHLY: "med-ogs-bio-rean-m",
        Frequency.YEARLY: "cmems_mod_med_bgc-bio_my_4.2km_P1Y-m",
    },
    ("nppv",): {
        Frequency.DAILY: "med-ogs-bio-rean-d",
        Frequency.MONTHLY: "med-ogs-bio-rean-m",
        Frequency.YEARLY: "cmems_mod_med_bgc-bio_my_4.2km_P1Y-m",
    },
}

GLOBAL_PRODUCTS = {
    # GLOBAL_MULTIYEAR_PHY_001_030
    ("thetao",): {
        Frequency.DAILY: "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        Frequency.MONTHLY: "cmems_mod_glo_phy_my_0.083deg_P1M-m",
    },
    (
        "uo",
        "vo",
    ): {
        Frequency.DAILY: "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        Frequency.MONTHLY: "cmems_mod_glo_phy_my_0.083deg_P1M-m",
    },
    ("mlotst",): {
        Frequency.DAILY: "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        Frequency.MONTHLY: "cmems_mod_glo_phy_my_0.083deg_P1M-m",
    },
    ("so",): {
        Frequency.DAILY: "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        Frequency.MONTHLY: "cmems_mod_glo_phy_my_0.083deg_P1M-m",
    },
}


class CMEMSProvider(Provider, ABC):
    @classmethod
    @abstractmethod
    def search_for_product(
        cls, var_name: VarName, frequency: Frequency
    ) -> str:
        """Given the name of a variable and a frequency, return the name of the
        CMEMS product that contains such a variable with the specified frequency.
        """
        raise NotImplementedError

    @classmethod
    def create(cls, config_file: Path | None = None) -> "Provider":
        if config_file is not None:
            raise ValueError(
                "CMEMS providers do not support configuration files"
            )
        return cls()

    def download_data(
        self,
        domain: Domain,
        frequency: Frequency,
        main_path: Path,
        data_files: Mapping[VarName, tuple[DataFile, ...]],
    ) -> None:
        """
        Downloads the data for the dataset.

        This method downloads all the missing data files for the dataset.
        If a file already exists, it will not be downloaded again.
        """
        for var_name, files in data_files.items():
            LOGGER.debug(
                'Downloading data for variable "%s": %s files will be '
                "downloaded",
                var_name,
                len(files),
            )

            product_id = self.search_for_product(
                var_name=var_name, frequency=frequency
            )
            LOGGER.debug(
                "The product associated with variable %s is: %s",
                var_name,
                product_id,
            )

            for file in files:
                # Ensure the path is relative to the main dir
                if not file.path.is_absolute():
                    file_path = main_path / file.path
                else:
                    file_path = file.path

                if file_path.exists():
                    LOGGER.debug(
                        'File "%s" already exists. Skipping download.',
                        file_path,
                    )
                    continue

                LOGGER.info('Downloading file "%s"', file_path)

                # Ensure the parent directory exists
                file_path.parent.mkdir(parents=True, exist_ok=True)

                output_filename = file_path.name
                temp_filename = "tmp." + output_filename
                temp_file_path = file_path.parent / temp_filename

                if temp_file_path.is_file():
                    LOGGER.debug(
                        '"%s" already exists. I will delete it!',
                        temp_file_path,
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
                    **domain.bounding_box.model_dump(exclude_none=True),
                )

                LOGGER.debug(
                    'Moving file "%s" to "%s"', temp_file_path, file_path
                )
                shutil.move(temp_file_path, file_path)


class CMEMSProviderMed(CMEMSProvider):
    @classmethod
    def get_name(cls) -> str:
        return "cmems_mediterranean"

    @classmethod
    def search_for_product(
        cls, var_name: VarName, frequency: Frequency
    ) -> str:
        """Given the name of a variable and a frequency, return the name of the
        CMEMS product that contains such a variable with the specified frequency.
        """
        selected_product = None
        for vars_tuple, prod_dict in MED_PRODUCTS.items():
            if var_name in vars_tuple:
                selected_product = prod_dict[frequency]
                break

        if selected_product is None:
            raise ValueError(
                f"Variable '{var_name}' is not available in the dictionary"
            )

        LOGGER.debug(
            f"var_name={var_name}, selected_product={selected_product}"
        )

        return selected_product

    def available_variables(self, frequency: Frequency) -> VariableDataset:
        variables_names = []
        for var_names, product_by_freq in MED_PRODUCTS.items():
            if frequency in product_by_freq:
                variables_names.extend(var_names)
        return VariableDataset(variables_names)


class CMEMSProviderGlobal(CMEMSProvider):
    @classmethod
    def get_name(cls) -> str:
        return "cmems_global"

    @classmethod
    def search_for_product(cls, var_name, frequency):
        selected_product = None
        for vars_tuple, prod_dict in GLOBAL_PRODUCTS.items():
            if var_name in vars_tuple:
                selected_product = prod_dict[frequency]
                break

        if selected_product is None:
            raise ValueError(
                f"Variable '{var_name}' is not available in the dictionary"
            )

        LOGGER.debug(
            f"var_name={var_name}, selected_product={selected_product}"
        )

        return selected_product

    def available_variables(self, frequency: Frequency) -> VariableDataset:
        variables_names = []
        for var_names, product_by_freq in GLOBAL_PRODUCTS.items():
            if frequency in product_by_freq:
                variables_names.extend(var_names)
        return VariableDataset(variables_names)
