import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from copernicusmarine import *  # noqa: F403


_COPERNICUSMARINE = None


def __getattr__(name: str):
    global _COPERNICUSMARINE

    if _COPERNICUSMARINE is None:
        import copernicusmarine

        # Remove all handlers associated with the copernicusmarine logger
        copernicusmarine_logger = logging.getLogger("copernicusmarine")
        copernicusmarine_logger.handlers.clear()

        _COPERNICUSMARINE = copernicusmarine

    return getattr(_COPERNICUSMARINE, name)
