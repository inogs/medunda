from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Mapping

from medunda.components.data_files import DataFile
from medunda.components.frequencies import Frequency
from medunda.domains.domain import Domain
from medunda.tools.typing import VarName


class Provider(ABC):
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def create(cls, config_file: Path | None = None) -> "Provider":
        raise NotImplementedError

    @abstractmethod
    def download_data(self, domain: Domain, frequency: Frequency, data_files: Mapping[VarName, tuple[DataFile, ...]]) -> None:
        raise NotImplementedError
