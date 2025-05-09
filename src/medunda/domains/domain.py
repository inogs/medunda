from pydantic import BaseModel


class Domain(BaseModel):
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float
    minimum_depth: float
    maximum_depth: float


GSA9 = Domain(
    minimum_latitude=41.29999921500007,
    maximum_latitude=44.42720294000003,
    minimum_longitude=7.525000098000021,
    maximum_longitude=13.003545339000027,
    minimum_depth=0,
    maximum_depth=250
)
