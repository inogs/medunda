from enum import Enum


class Frequency(Enum):
    DAILY = "daily"
    MONTHLY = "monthly"

    def __str__(self) -> str:
        if self == Frequency.DAILY:
            return "daily"
        elif self == Frequency.MONTHLY:
            return "monthly"
        else:
            raise ValueError("Invalid Frequency value")
