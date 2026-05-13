_CMOCEAN = None


def __getattr__(name: str):
    global _CMOCEAN
    if _CMOCEAN is None:
        import cmocean as _CMOCEAN
    return getattr(_CMOCEAN, name)


def get_cmocean_map(name: str):
    global _CMOCEAN
    if _CMOCEAN is None:
        import cmocean as _CMOCEAN
    return getattr(_CMOCEAN.cm, name)
