from pydantic import BaseModel


class Domain(BaseModel):
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float
    minimum_depth: float
    maximum_depth: float


GSA9 = Domain(
    minimum_latitude=41,
    maximum_latitude=44,
    minimum_longitude=9,
    maximum_longitude=13,
    minimum_depth=0,
    maximum_depth=250
)
