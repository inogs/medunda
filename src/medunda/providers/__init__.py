from medunda.providers.cmems import CMEMSProvider
from medunda.providers.provider import Provider


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
    providers: dict[str, type[Provider]] = {
        CMEMSProvider.get_name(): CMEMSProvider,
        # Add other providers here as needed
    }

    if name not in providers:
        raise ValueError(f"Provider '{name}' is not recognized.")

    return providers[name]
