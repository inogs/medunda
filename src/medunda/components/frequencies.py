from enum import Enum


class Frequency(Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"

    def __str__(self) -> str:
        if self == Frequency.DAILY:
            return "daily"
        elif self == Frequency.MONTHLY:
            return "monthly"
        elif self == Frequency.YEARLY:
            return "yearly"
        elif self == Frequency.WEEKLY:
            return "weekly"
        else:
            raise ValueError("Invalid Frequency value")
