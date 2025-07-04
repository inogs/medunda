from datetime import datetime
from logging import getLogger
from pathlib import Path
import re
import shutil

import copernicusmarine
from pydantic import BaseModel


from medunda.domains.domain import Domain
from medunda.sources.cmems import search_for_product
from medunda.tools.file_names import from_file_path_to_time_range
from medunda.tools.typing import VarName


LOGGER = getLogger(__name__)


class Dataset(BaseModel):
    """
    A class to represent a dataset for a specific domain.
    
    Attributes:
        domain: The domain associated with the dataset.
        start_date: The start date of the dataset.
        end_date: The end date of the dataset.
    """
    domain: Domain
    start_date: datetime
    end_date: datetime
    data_files: dict[VarName, tuple[Path, ...]] = {}
    frequency: str = "monthly"
    source: str = "cmems"

    def download_data(self):
        """
        Downloads the data for the dataset.
        
        This method downloads all the missing data files for the dataset.
        If a file already exists, it will not be downloaded again.
        """
        for var_name, files in self.data_files.items():
            for file in files:
                if file.exists():
                    LOGGER.debug(
                        f'File "%s" already exists. Skipping download.', file
                    )
                    continue
                
                # Ensure the parent directory exists
                file_path = file.absolute()
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

                product_id = search_for_product(
                    var_name=var_name,
                    frequency=self.frequency
                )
                start, end = from_file_path_to_time_range(file_path)
                copernicusmarine.subset(
                    dataset_id=product_id,
                    variables=[var_name],
                    start_datetime=start,
                    end_datetime=end,
                    output_filename=str(temp_file_path),
                    **self.domain.model_dump(exclude={"name"})
                )

                shutil.move(temp_file_path, file_path)

