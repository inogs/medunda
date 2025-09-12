from collections.abc import Collection
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar

import cmocean
from matplotlib.colors import Colormap

from medunda.tools.typing import VarName

DEFAULT_COLORMAP = Colormap("viridis")


@dataclass(frozen=True)
class Variable:
    """A class representing a variable with associated metadata.

    This class manages variables with unique names, labels, and colormaps. It maintains
    a registry of all created variables to ensure name uniqueness and allow retrieval
    by name.

    Attributes:
        name: The unique identifier for the variable.
        label: A human-readable label for the variable. If None, the name is used.
        cmap: The colormap specification. Can be a string name, a Colormap object,
            or None (uses default colormap).

        _variables (ClassVar[dict[VarName, "Variable"]]): Class-level registry of all
            variables.

    Methods:
        get_colormap(): Returns the Colormap object for this variable.
        get_label(): Returns the display label for this variable.

        get_by_name(name): Class method to retrieve a variable by its name.

    Raises:
        ValueError: If attempting to create a variable with a name that already exists,
            or when trying to retrieve a non-existent variable.
    """
    name: VarName
    label: str | None = None
    cmap: str | Colormap | None = None

    _variables: ClassVar[dict[VarName, "Variable"]] = {}

    def get_colormap(self) -> Colormap:
        if self.cmap is None:
            return DEFAULT_COLORMAP
        if isinstance(self.cmap, str):
            return Colormap(self.cmap)
        return self.cmap

    def get_label(self) -> str:
        if self.label is None:
            return self.name
        return self.label

    def __post_init__(self):
        cls = self.__class__
        if self.name in cls._variables:
            raise ValueError(
                f"Variable with name '{self.name}' already exists"
            )
        cls._variables[self.name] = self

    @classmethod
    def get_by_name(cls, name: VarName) -> "Variable":
        if name not in cls._variables:
            raise ValueError(f"Variable with name '{name}' not found")
        return cls._variables[name]


Variable("thetao", label="Potential Temperature", cmap=cmocean.cm.thermal)  # pyright: ignore[reportAttributeAccessIssue]
Variable("so", label="Practical Salinity", cmap="viridis")
Variable("o2", label="Dissolved Oxygen", cmap=cmocean.cm.deep)  # pyright: ignore[reportAttributeAccessIssue]
Variable("chl", label="Chlorophyll-a", cmap=cmocean.cm.algae)  # pyright: ignore[reportAttributeAccessIssue]
Variable("nppv", label="Net Primary Production", cmap=cmocean.cm.matter)  # pyright: ignore[reportAttributeAccessIssue]
Variable("uo", label="Zonal Velocity", cmap=cmocean.cm.balance)  # pyright: ignore[reportAttributeAccessIssue]
Variable("vo", label="Meridional Velocity", cmap=cmocean.cm.balance) # pyright: ignore[reportAttributeAccessIssue]
Variable("ph", label="pH", cmap=cmocean.cm.balance)  # pyright: ignore[reportAttributeAccessIssue]
Variable("no3", label="Nitrate", cmap=cmocean.cm.matter)  # pyright: ignore[reportAttributeAccessIssue]
Variable("po4", label="Phosphate", cmap=cmocean.cm.matter)  # pyright: ignore[reportAttributeAccessIssue]
Variable("si", label="Silicate", cmap=cmocean.cm.matter)  # pyright: ignore[reportAttributeAccessIssue]


class VariableDataset(Collection[Variable]):
    """A collection of variables that can be used to store and manage multiple Variable objects.

    This class implements the Collection interface and provides methods to store, retrieve,
    and manage Variable objects. Variables can be added individually or loaded from their names.

    Args:
        variables (Iterable[VarName] | None, optional): An iterable of variable names to initialize
            the dataset with. If None, creates an empty dataset. Defaults to None.

    Methods:
        add_variable(var: Variable) -> None: Adds a Variable object to the dataset.
        get_variable_names() -> set[VarName]: Returns a set of all variable names in the dataset.

        all_variables() -> VariableDataset: Creates a new dataset containing all registered variables.

    Implements:
        __contains__(item: object) -> bool: Checks if a variable or variable name is in the dataset.
        __len__() -> int: Returns the number of variables in the dataset.
        __iter__(): Returns an iterator over the Variable objects in the dataset.
    """
    def __init__(self, variables: Iterable[VarName] | None = None) -> None:
        if variables is None:
            self._variables = {}
        else:
            self._variables: dict[VarName, Variable] = {
                name: Variable.get_by_name(name) for name in variables
            }

    def __contains__(self, item: object) -> bool:
        if item in self._variables:
            return True

        if not isinstance(item, Variable):
            return False

        return item.name in self._variables

    def __len__(self) -> int:
        return len(self._variables)

    def __iter__(self):
        return iter(self._variables.values())

    def add_variable(self, var: Variable) -> None:
        self._variables[var.name] = var

    def get_variable_names(self) -> set[VarName]:
        return set(self._variables.keys())

    @classmethod
    def all_variables(cls) -> "VariableDataset":
        return cls(variables=Variable._variables.keys())
