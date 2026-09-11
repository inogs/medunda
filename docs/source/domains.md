(domaindoc)=

# Domains

A **domain** defines the geographical and, optionally, vertical extent of the data to be retrieved or processed by MEDUNDA.
Domains are used to restrict oceanographic datasets to a specific geographical area. A domain can be defined as a rectangular area, a polygon, or a collection of polygons.
A domain can also optionally include minimum and maximum depth constraints.

```{eval-rst}
.. autoclass:: medunda.domains.domain.Domain
   :members:
   :show-inheritance:
```

Medunda supports rectangular, polygonal, and multipolygonal domains. Domains can be defined directly in Python or loaded from external geographic definitions such as shapefiles, WKT files, and BitSea basins.

## RectangularDomain

A rectangular domain is defined by minimum and maximum latitude and longitude values. It can optionally include minimum and maximum depth constraints.

```{eval-rst}
.. autoclass:: medunda.domains.domain.RectangularDomain
   :members:
   :show-inheritance:
```

Example:
```yaml
---
name: AdriaticSea
geometry:
  type: Rectangle
  min_latitude: 39.609258912544874
  max_latitude: 45.78273415600006
  min_longitude: 12.154220053000188
  max_longitude: 20.19036466397057
depth:
  min_depth: 1.0182366371154785
  max_depth: 1250
```

## PolygonalDomain

```{eval-rst}
.. autoclass:: medunda.domains.domain.PolygonalDomain
   :members:
   :show-inheritance:
```

Example: North Adriatic Sea:

```text
WKT,nome,descrizione
"POLYGON ((12.0548867 44.9596823, 12.4751137 44.6146529, 13.8456582 44.6224729, 14.2329262 44.9091292, 14.0846108 45.7931288, 13.439164 45.9308423, 11.9010781 45.6281905, 12.0548867 44.9596823))",NorthAdr,
```

## MultiPolygonalDomain

```{eval-rst}
.. autoclass:: medunda.domains.domain.MultiPolygonalDomain
   :members:
   :show-inheritance:
```

## domain_from_basin

Creates a domain from a predefined BitSea basin identified by its UUID. This allows basin definitions to be directly incorporated into Medunda workflows.

```{eval-rst}
.. autofunction:: medunda.domains.domain.domain_from_basin
```

Example:

```yaml
---
name: CentralMediterranean
geometry:
  type: basin
  uuid: mid3
```

## read_zipped_shapefile

Reads a shapefile stored in a ZIP archive and extracts the selected feature. The resulting geometry can be used to define a domain.

```{eval-rst}
.. autofunction:: medunda.domains.domain.read_zipped_shapefile
```

Example:

```yaml
---
name: GSA9
geometry:
  type: shapefile
  file_path: "${MAIN_DIR}/data/GSAs_simplified.zip"
  selection_field_name: SECT_COD
  selection_field_value: GSA09
depth:
  min_depth: 1.0182366371154785
  max_depth: 2961
```

## read_domain_from_yaml

Creates a domain from a YAML configuration file. The definition can specify the geometry and optional depth constraints.

```{eval-rst}
.. autofunction:: medunda.domains.domain.read_domain_from_yaml
```

## domain_from_string

Creates a domain from its string representation. It provides a convenient way to construct domains from textual specifications.

```{eval-rst}
.. autofunction:: medunda.domains.domain.domain_from_string
```
