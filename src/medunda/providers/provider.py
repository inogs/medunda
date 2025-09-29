from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Mapping

from medunda.components.data_files import DataFile
from medunda.components.frequencies import Frequency
from medunda.components.variables import VariableDataset
from medunda.domains.domain import Domain
from medunda.tools.typing import VarName


class Provider(ABC):
    """
    A source from which a `Domain` can download its data

    Medunda can download the data from different sources: from tar archives,
    from the copernicusmarine data store, etc...
    To each Medunda dataset is associated a `Provider`, that takes care of
    downloading the necessary data from an external database.

    This class is an abstract class that defines the interface that all the
    providers must implement.
    """
    @classmethod
    def get_name(cls) -> str:
        """
        Return the name of this Provider
        """
        return cls.__name__

    @classmethod
    @abstractmethod
    def create(cls, config_file: Path | None = None) -> "Provider":
        """
        Creates a new `Provider` instance from an optional config file

        Some Providers require initialization parameters. To handle these
        parameters, a `Provider` object can be instantiated using the
        `create` method that accepts a single argument: the path to a YAML
        configuration file.

        Each provider defines its own configuration file fields. Not all
        providers need a configuration file: this is typically the case when
        a provider either has default values for all its options or doesn't
        require any initialization options. In such cases, the config_file
        argument can be `None`.

        Args:
            config_file: Path to the configuration file. Can be `None` if the
                concrete implementation of the Provider doesn't require
                initialization values

        Returns:
            A new `Provider` instance
        """
        raise NotImplementedError

    @abstractmethod
    def download_data(
            self,
            domain: Domain,
            frequency: Frequency,
            main_path: Path,
            data_files: Mapping[VarName, tuple[DataFile, ...]]
        ) -> None:
        """
        Downloads the required data files.

        Args:
            domain: The domain for which the files need to be downloaded.
            frequency: The temporal resolution of the data to download.
            main_path: The root directory where the files will be saved.
            data_files: A mapping of each variable to the files that need to
                be downloaded for that variable. If a `DataFile` contains a
                relative path, it is interpreted as relative to `main_path`.
        """
        raise NotImplementedError

    def available_variables(self, frequency: Frequency) -> VariableDataset:
        """
        Returns all variables available for the given frequency

        Not all variables are available from each provider at every frequency.
        Some variables may only be available at specific frequencies, while
        others might not be available at all. This method returns a set of
        variables that are available from this `Provider` for the specified
        frequency.

        Args:
            frequency: The frequency for which to check variable availability

        Returns:
            A `VariableDataset` containing all available variables
        """
        return VariableDataset.all_variables()
