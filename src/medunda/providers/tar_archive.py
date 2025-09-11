import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import yaml

from medunda.components.data_files import DataFile
from medunda.components.frequencies import Frequency
from medunda.components.variables import Variable
from medunda.components.variables import VariableDataset
from medunda.domains.domain import Domain
from medunda.providers.provider import Provider
from medunda.tools.typing import VarName


LOGGER = logging.getLogger(__name__)


class TarArchiveProvider(Provider):
    """Provider for tar archives."""

    def __init__(self, name: str,
                 variables: dict[VarName, dict[str, str]],
                 frequencies: dict[Frequency, dict],
                 source: str = "local",
                 start_time: datetime | None = None,
                 end_time: datetime | None = None):
        self.name = name
        self._variables = variables
        self._frequencies = frequencies
        self._source = source
        self._start_time = start_time
        self._end_time = end_time

    @classmethod
    def get_name(cls) -> str:
        return "tar_archive"

    def available_variables(self, frequency: Frequency) -> VariableDataset:
        if frequency not in self._frequencies:
            return VariableDataset()
        return VariableDataset(self._frequencies[frequency]["variables"])


    def download_data(self, domain: Domain, frequency: Frequency, data_files: Mapping[VarName, tuple[DataFile, ...]]) -> None:
        for var_name, files in data_files.items():
            if var_name not in self._variables:
                raise ValueError(
                    f'Variable "{var_name}" is not defined in the provider '
                    f'"{self.name}".'
                )

        LOGGER.error("The download_data method is not implemented yet.")

    @staticmethod
    def _read_frequencies_in_config(config_content: dict, variables: dict[VarName, dict]) -> dict[Frequency, dict]:
        output: dict[Frequency, dict] = {}
        if "frequencies" not in config_content:
            raise ValueError(
                'The configuration file must contain a "frequencies" section.'
            )
        if not isinstance(config_content["frequencies"], list):
            raise ValueError(
                'The "frequencies" section must be a list'
            )

        required_keys = {"frequency", "variables", "path"}
        for frequency in config_content["frequencies"]:
            if not isinstance(frequency, Mapping):
                raise ValueError(
                    'Each frequency in the "frequencies" section must be a '
                    "dictionary with the following keys: {}. "
                    "Received: {}".format(
                        ", ". join([f for f in sorted(list(required_keys))]),
                        frequency
                    )
                )

            missing_keys = required_keys - set(frequency.keys())
            if missing_keys:
                raise ValueError(
                    'Each frequency in the "frequencies" section must contain '
                    "the following keys: {}. Missing keys in {} ---> {}".format(
                        ", ". join([f for f in sorted(list(required_keys))]),
                        frequency,
                        ", ".join([f for f in sorted(list(missing_keys))])
                    )
                )

            for key in frequency.keys():
                if key not in required_keys:
                    raise ValueError(
                        f'Invalid key "{key}" in frequency {frequency}. '
                        f'Supported keys are: {", ". join([f for f in sorted(list(required_keys))])}.'
                    )

            try:
                frequency_value = Frequency(frequency["frequency"].lower())
            except Exception as e:
                raise ValueError(
                    f"Invalid frequency '{frequency['frequency']}' in the "
                    f"section {frequency}. Supported frequencies are 'monthly' "
                    "and 'daily'."
                ) from e

            frequency_path = Path(frequency["path"])
            frequency_vars = frequency["variables"]
            if not isinstance(frequency_vars, list):
                raise ValueError(
                    f'The "variables" key in frequency {frequency} must be a list. Received: {frequency_vars}'
                )
            for var in frequency_vars:
                if var not in variables:
                    raise ValueError(
                        f'Variable "{var}" in frequency {frequency} is not defined in the "variables" section.'
                    )

            if frequency_value in output:
                raise ValueError(
                    f"Frequency '{frequency_value}' is defined multiple times."
                )

            output[frequency_value] = {
                "path": frequency_path,
                "variables": frequency_vars
            }

        return output


    @staticmethod
    def _read_variables_in_config(config_content: dict) -> dict[VarName, dict[str, str]]:
        variables_config: dict[VarName, dict[str, str]] = {}
        if "variables" not in config_content:
            raise ValueError(
                'The configuration file must contain a "variables" section.'
            )
        if not isinstance(config_content["variables"], list):
            raise ValueError(
                'The "variables" section must be a list.'
            )

        for raw_var in config_content["variables"]:
            if not isinstance(raw_var, Mapping):
                raise ValueError(
                    'Each variable in the "variables" section must be a '
                    "dictionary with the following mandatory keys: "
                    "variable and unit. Optional keys are: dataset_name. "
                    f"Received: {raw_var}"
                )
            if "variable" not in raw_var or "unit" not in raw_var:
                raise ValueError(
                    'Each variable in the "variables" section must contain '
                    'the keys "variable" and "unit". Optional keys are: '
                    f'dataset_name. Received: {raw_var}'
                )
            for key in raw_var.keys():
                if key not in {"variable", "unit", "dataset_name"}:
                    raise ValueError(
                        f'Invalid key "{key}" in variable {raw_var}. '
                        'Supported keys are: "variable", "unit", and '
                        '"dataset_name".'
                    )
            var_name = raw_var["variable"]
            try:
                current_var = Variable.get_by_name(var_name)
            except ValueError as e:
                raise ValueError(
                    f'Invalid variable "{var_name}" in the "variables" '
                    "section. Medunda does not support this variable."
                ) from e

            variables_config[var_name] = {
                "unit": raw_var["unit"],
                "dataset_name": raw_var.get("dataset_name", current_var.name)
            }

        return variables_config


    @classmethod
    def create(cls, config_file: Path | None = None) -> "Provider":
        if config_file is None:
            raise ValueError(
                f"A {cls.__name__} configuration file must be provided."
            )

        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file {config_file} does not exist."
            )

        config_content = yaml.safe_load(config_file.read_text())

        variables = cls._read_variables_in_config(config_content)

        # Validate the frequencies
        frequencies = cls._read_frequencies_in_config(config_content, variables)

        return TarArchiveProvider(
            name=config_content.get("name", "tar_archive"),
            variables=variables,
            frequencies=frequencies,
            source=config_content.get("source", "local"),
            start_time=config_content.get("start_time", None),
            end_time=config_content.get("end_time", None)
        )
