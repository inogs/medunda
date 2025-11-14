import os
from pathlib import Path
from tempfile import TemporaryDirectory as _TemporaryDirectory


class TemporaryDirectory:
    """Temporary directory context manager for Medunda tools.

    This class provides a context manager for creating and managing temporary
    directories. It ensures that the directory is created upon entering the
    context and removed upon exiting.

    It is basically a wrapper around the built-in TemporaryDirectory class, but
    there are two main differences: the context manager returns a Path object
    instead of a string, and the directory is created inside the scratch dir
    if an environment variable SCRATCH is set.

    The idea of using a scratch directory is to avoid issues on clusters where
    the /tmp directory is not shared with all nodes (and it is usually very
    small). This class ensures that all the temporary directories are created
    inside a directory named `.medunda_temp` which is created inside the scratch
    directory.
    """

    def __init__(self):
        self._temp_dir: _TemporaryDirectory | None = None
        self._scratch_path = None
        scratch_dir = os.getenv("SCRATCH")
        if scratch_dir is not None:
            self._scratch_path = Path(scratch_dir) / ".medunda_temp"
            self._scratch_path.mkdir(exist_ok=True)

    def __enter__(self):
        temp_dir_kwargs = {}
        if self._scratch_path is not None:
            temp_dir_kwargs["dir"] = self._scratch_path
        self._temp_dir = _TemporaryDirectory(**temp_dir_kwargs)
        return Path(self._temp_dir.__enter__())

    def __exit__(self, exc_type, exc_value, traceback):
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
        self._temp_dir = None
