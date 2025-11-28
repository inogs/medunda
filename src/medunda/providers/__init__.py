from collections.abc import Mapping
from types import MappingProxyType

from medunda.providers.cmems import CMEMSProviderGlobal
from medunda.providers.cmems import CMEMSProviderMed
from medunda.providers.provider import Provider
from medunda.providers.tar_archive import TarArchiveProvider

PROVIDERS: Mapping[str, type[Provider]] = MappingProxyType(
    {
        CMEMSProviderGlobal.get_name(): CMEMSProviderGlobal,
        CMEMSProviderMed.get_name(): CMEMSProviderMed,
        TarArchiveProvider.get_name(): TarArchiveProvider,
        # Add other providers here as needed
    }
)


def get_provider(name: str) -> type[Provider]:
    """
    Returns the provider class associated with the given name.

    Args:
        name: The name of the provider.

    Returns:
        The provider class associated with the given name.

    Raises:
        ValueError: If the provider name is not recognized.
    """
    if name not in PROVIDERS:
        raise ValueError(f"Provider '{name}' is not recognized.")

    return PROVIDERS[name]
