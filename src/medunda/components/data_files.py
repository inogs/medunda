from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from medunda.tools.typing import VarName


class DataFile(BaseModel):
    start_date: datetime
    end_date: datetime
    variable: VarName
    path: Path
