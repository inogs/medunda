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

    This class manages variables with unique names, labels, and colormaps.
    It maintains a registry of all created variables to ensure name uniqueness
    and allow retrieval by name.

    Attributes:
        name: The unique identifier for the variable.
        label: A human-readable label for the variable. If None, the name is
            used.
        cmap: The colormap specification. Can be a string name, a Colormap
            object, or None (uses default colormap).

    Raises:
        ValueError: If attempting to create a variable with a name that already exists,
            or when trying to retrieve a non-existent variable.
    """

    name: VarName
    label: str | None = None
    cmap: str | Colormap | None = None

    _variables: ClassVar[dict[VarName, "Variable"]] = {}

    def get_colormap(self) -> Colormap:
        """
        Returns the colormap associated to this variable.

        Calling this method is safer than reading directly the `.cmap`
        attribute, because this method takes care of returning a default
        value when the `cmap` attribute is `None`.

        Returns:
            A colormap associated to the variable
        """
        if self.cmap is None:
            return DEFAULT_COLORMAP
        if isinstance(self.cmap, str):
            return Colormap(self.cmap)
        return self.cmap

    def get_label(self) -> str:
        """
        Return a label that identifies a variable

        This method returns a human-name for tor the variable. If this
        name has not been submitted during the initialization of the
        variable, it simply returns its name.

        Returns:
            The label of the variable
        """
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
        """
        Returns a Variable from its name.

        Given the name of a variable, this method checks if a variable
        with this name has ever been initialized and, if this is the
        case, returns its object.

        Returns:
            A Variable object whose name coincides with the passed argument

        Raises:
            ValueError if there is no variable with the requested name
        """
        if name not in cls._variables:
            raise ValueError(f"Variable with name '{name}' not found")
        return cls._variables[name]


# Here there is a list of all the variables defined inside Medunda. If you
# want to add new variables, you must add them here. This part of the code
# also defines the official name that Medunda uses to refer to those
# variables.

# This prevents pyright from reporting attribute access issues about the
# cmocean colormaps
# pyright: reportAttributeAccessIssue=false

Variable("thetao", label="Potential Temperature", cmap=cmocean.cm.thermal)
Variable(
    "bottomT",
    label="Sea water potential temperature at sea floor",
    cmap=cmocean.cm.thermal,
)
Variable("so", label="Practical Salinity", cmap="viridis")
Variable("o2", label="Dissolved Oxygen", cmap=cmocean.cm.deep)
Variable("chl", label="Chlorophyll-a", cmap=cmocean.cm.algae)
Variable("nppv", label="Net Primary Production", cmap=cmocean.cm.matter)
Variable("uo", label="Zonal Velocity", cmap=cmocean.cm.balance)
Variable("vo", label="Meridional Velocity", cmap=cmocean.cm.balance)
Variable("mlotst", label="Mixed Layer Depth", cmap=cmocean.cm.deep)
Variable("ph", label="pH", cmap=cmocean.cm.balance)
Variable("no3", label="Nitrate", cmap=cmocean.cm.matter)
Variable("po4", label="Phosphate", cmap=cmocean.cm.matter)
Variable("si", label="Silicate", cmap=cmocean.cm.matter)
Variable("B1c", label="Aerobic and Anaerobic Bacteria carbon", cmap="viridis")
Variable(
    "B1n", label="Aerobic and Anaerobic Bacteria nitrogen", cmap="viridis"
)
Variable(
    "B1p", label="Aerobic and Anaerobic Bacteria phosphorous", cmap="viridis"
)
Variable("P1c", label="Diatoms Biomass_carbon", cmap=cmocean.cm.matter)
Variable("P2c", label="Flagellates Biomass_carbon", cmap=cmocean.cm.matter)
Variable(
    "P3c", label="PicoPhytoplankton Biomass_carbon", cmap=cmocean.cm.matter
)
Variable(
    "P4c", label="Large Phytoplankton Biomass_carbon", cmap=cmocean.cm.matter
)
Variable("Z3c", label="Carnivorous Mesozooplankton_carbon", cmap="viridis")
Variable("Z4c", label="Omnivorous Mesozooplankton_carbon", cmap="viridis")
Variable("Z5c", label="Microzooplankton_carbon", cmap="viridis")
Variable("Z6c", label="Heterotrophic Nanoflagellates", cmap="viridis")

Variable("R1c", label="Dissolved Organic Carbon", cmap=cmocean.cm.matter)
Variable(
    "R2c", label="Semi-labile Dissolved Organic Carbon", cmap=cmocean.cm.matter
)
Variable(
    "R3c",
    label="Semi-refactory Dissolved Organic Carbon",
    cmap=cmocean.cm.matter,
)
Variable("R6c", label="Particulate Organic Carbon", cmap=cmocean.cm.matter)


class VariableDataset(Collection[Variable]):
    """A Collection of `Variables`.

    A `VariableDataset` is a collection of variables that can be used to
    store and manage multiple Variable objects.

    This class implements the Collection interface and provides methods to
    store, retrieve, and manage Variable objects. Variables can be added
    individually or loaded from their names.

    Args:
        variables: An iterable of variable names to initialize
            the dataset with. If not submitted, an empty dataset is
            created
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
        """
        Adds a Variable object to the dataset.
        """
        self._variables[var.name] = var

    def get_variable_names(self) -> set[VarName]:
        """
        Returns a set of all variable names in the dataset.
        """
        return set(self._variables.keys())

    @classmethod
    def all_variables(cls) -> "VariableDataset":
        """
        Creates a new dataset containing all registered variables.

        Return a `VariableDataset` that contains all the variables defined
        inside Medunda. This method checks all the `Variable` that are
        instantiated, so it must be called after having defined the variables.
        """
        return cls(variables=Variable._variables.keys())
