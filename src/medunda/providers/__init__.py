from collections.abc import Mapping
from types import MappingProxyType

from medunda.providers.cmems import CMEMSProvider
from medunda.providers.tar_archive import TarArchiveProvider
from medunda.providers.provider import Provider


PROVIDERS: Mapping[str, type[Provider]] = MappingProxyType({
    CMEMSProvider.get_name(): CMEMSProvider,
    TarArchiveProvider.get_name(): TarArchiveProvider,
    # Add other providers here as needed
})


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
