(domaindoc)=

# Doamins

A **domain** defines the geographical and, optionally, vertical extent of the data to be retrieved or processed by MEDUNDA.
Domains are used to restrict oceanographic datasets to a specific geographical area. A domain can be defined as a rectangular area, a polygon, or a collection of polygons.
A domain can also optionally include minimum and maximum depth constraints.

Medunda supports rectangular, polygonal, and multipolygonal domains. Domains can be defined directly in Python or loaded from external geographic definitions such as shapefiles, WKT files, and BitSea basins.


---

## Domain

```{eval-rst}
.. autoclass:: medunda.domains.domain.Domain
   :members:
   :show-inheritance:
```

---

## RectangularDomain

```{eval-rst}
.. autoclass:: medunda.domains.domain.RectangularDomain
   :members:
   :show-inheritance:
```

---

## PolygonalDomain

```{eval-rst}
.. autoclass:: medunda.domains.domain.PolygonalDomain
   :members:
   :show-inheritance:
```

---

## MultiPolygonalDomain

```{eval-rst}
.. autoclass:: medunda.domains.domain.MultiPolygonalDomain
   :members:
   :show-inheritance:
```

---

## domain_from_basin

```{eval-rst}
.. autofunction:: medunda.domains.domain.domain_from_basin
```

---

## read_zipped_shapefile

```{eval-rst}
.. autofunction:: medunda.domains.domain.read_zipped_shapefile
```

---

## read_domain_from_yaml

```{eval-rst}
.. autofunction:: medunda.domains.domain.read_domain_from_yaml
```

---

## domain_from_string

```{eval-rst}
.. autofunction:: medunda.domains.domain.domain_from_string
```
