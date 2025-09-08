from collections.abc import Collection
from dataclasses import dataclass
from typing import ClassVar


import cmocean
from matplotlib.colors import Colormap

from medunda.tools.typing import VarName


DEFAULT_COLORMAP = Colormap("viridis")


@dataclass(frozen=True)
class Variable:
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
Variable("nppv", label="Net Primary Production", cmap=cmocean.cm.matter)  # pyright: ignore[reportAttributeAccessIssue]


class VariableDataset(Collection[Variable]):
    def __init__(self, variables: list[VarName] | None = None) -> None:
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
