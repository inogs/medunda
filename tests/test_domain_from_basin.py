import numpy as np
import xarray as xr

from medunda.domains.domain import MultiPolygonalDomain
from medunda.domains.domain import PolygonalDomain
from medunda.domains.domain import domain_from_basin


class FakeSimpleBasin:
    def __init__(self, uuid, name, borders):
        self._uuid = uuid
        self.name = name
        self.borders = borders

    def get_uuid(self):
        return self._uuid


class FakeComposedBasin:
    def __init__(self, uuid, name, basin_list):
        self._uuid = uuid
        self.name = name
        self.basin_list = basin_list

    def get_uuid(self):
        return self._uuid


def test_domain_from_basin_expands_nested_composed_basins(monkeypatch):
    basin_1 = FakeSimpleBasin(
        uuid="simple.1",
        name="simple1",
        borders=[(0.0, 0.0), (1.0, 0.0), (0.5, 1.0)],
    )
    basin_2 = FakeSimpleBasin(
        uuid="simple.2",
        name="simple2",
        borders=[(2.0, 0.0), (3.0, 0.0), (2.5, 1.0)],
    )
    nested = FakeComposedBasin(
        uuid="composed.nested", name="nested", basin_list=[basin_2]
    )
    root = FakeComposedBasin(
        uuid="composed.root",
        name="root",
        basin_list=[basin_1, nested],
    )

    def fake_load_from_uuid(uuid):
        assert uuid == "composed.root"
        return root

    monkeypatch.setattr(
        "medunda.domains.domain.bitsea_basin.Basin.load_from_uuid",
        fake_load_from_uuid,
    )

    domain = domain_from_basin("composed.root", name="my-domain")

    assert isinstance(domain, MultiPolygonalDomain)
    assert domain.name == "my-domain"
    assert {polygon.name for polygon in domain.polygons} == {
        "simple1",
        "simple2",
    }


def test_multi_polygonal_domain_selection_mask_is_union_of_subdomains():
    dataset = xr.Dataset(
        coords={
            "latitude": np.array([0.0, 1.0], dtype=float),
            "longitude": np.array([0.0, 1.0], dtype=float),
        }
    )

    polygon_1 = PolygonalDomain.create_from_coordinates(
        name="poly1",
        longitudes=[0.0, 1.0, 0.0],
        latitudes=[0.0, 0.0, 1.0],
    )
    polygon_2 = PolygonalDomain.create_from_coordinates(
        name="poly2",
        longitudes=[0.0, 1.0, 1.0],
        latitudes=[1.0, 0.0, 1.0],
    )
    domain = MultiPolygonalDomain.create_from_polygons(
        name="multi", polygons=[polygon_1, polygon_2]
    )

    expected = np.logical_or(
        polygon_1.compute_selection_mask(dataset),
        polygon_2.compute_selection_mask(dataset),
    )
    result = domain.compute_selection_mask(dataset)
    np.testing.assert_array_equal(result, expected)
