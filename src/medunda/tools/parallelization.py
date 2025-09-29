import os
from logging import getLogger
from multiprocessing import cpu_count
from socket import gethostname


LOGGER = getLogger(__name__)

PROCESSES_VAR_NAME = "MedundaPROCS"


def _read_procs_from_var(var_name) -> int | None:
    n_processes_var = os.environ.get(var_name)

    if n_processes_var is not None:
        LOGGER.debug(
            "Variable %s is defined and has value %s",
            var_name,
            n_processes_var
        )

        try:
            n_processes = int(n_processes_var)
        except ValueError as e:
            raise ValueError(
                f'Variable "{var_name}" is defined but its value '
                f'("{n_processes_var}") is not an integer'
            ) from e
        LOGGER.debug("Running with %s processes", n_processes)
        return n_processes

    LOGGER.debug("Variable %s is not defined", var_name)

    return None



def get_n_of_processes() -> int:
    LOGGER.debug("Counting how many processors we must use")

    n_processes = _read_procs_from_var(PROCESSES_VAR_NAME)
    if n_processes is not None:
        LOGGER.debug(
            "Running with %i processors (defined by %s)",
            n_processes,
            PROCESSES_VAR_NAME
        )
        return n_processes

    n_processes = _read_procs_from_var("SLURM_TASKS_PER_NODE")
    if n_processes is not None:
        LOGGER.debug(
            "Running with %i processors (defined by %s)",
            n_processes,
            "SLURM_TASKS_PER_NODE"
        )
        return n_processes

    n_processes = cpu_count()

    hpc_system = os.environ.get("HPC_SYSTEM")
    running_on_cineca = False
    if hpc_system is not None:
        LOGGER.debug(
            "HPC_SYSTEM env variable is defined as %s", hpc_system
        )
        if hpc_system in ("g100", "leonardo"):
            running_on_cineca = True
    else:
        LOGGER.debug("HPC_SYSTEM is not defined")

    LOGGER.debug("Are we running on CINECA? %s", running_on_cineca)
    if running_on_cineca:
        hostname = gethostname()
        LOGGER.debug("Current hostname is %s", hostname)
        if hostname.startswith("login"):
            LOGGER.debug(
                "Decreasing the number of workers because we are running on "
                "a login node"
                )
            n_processes = min(n_processes, 4)

    return n_processes
