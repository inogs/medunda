(actionsdoc)=
# Actions

Actions are the processing operations that can be applied to oceanographic datasets in Medunda.
Each action is exposed as a sub-command of the CLI and as a Python function that accepts an
`xarray.Dataset` and returns a transformed dataset.

---

## average\_between\_layers

```{eval-rst}
.. autofunction:: medunda.actions.average_between_layers.average_between_layers
```

---

## calculate\_stats

```{eval-rst}
.. autoclass:: medunda.actions.calculate_stats.Stats
   :members:
   :show-inheritance:
```

```{eval-rst}
.. autofunction:: medunda.actions.calculate_stats.calculate_stats
```

---

## climatology

```{eval-rst}
.. autofunction:: medunda.actions.climatology.climatology
```

---

## reduce\_axes

```{eval-rst}
.. autofunction:: medunda.actions.reduce_axes.reduce_axes
```

---

## compute\_integral

```{eval-rst}
.. autofunction:: medunda.actions.integration.compute_integral
```

---

## extract\_annual\_extremes

```{eval-rst}
.. autofunction:: medunda.actions.extract_annual_extremes.extract_annual_extremes
```

---

## extract\_annual\_extremes\_per\_layer

```{eval-rst}
.. autofunction:: medunda.actions.extract_annual_extremes_per_layer.extract_annual_extremes_per_layer
```

---

## extract\_bottom

```{eval-rst}
.. autofunction:: medunda.actions.extract_bottom.extract_bottom
```

---

## extract\_layer

```{eval-rst}
.. autofunction:: medunda.actions.extract_layer.extract_layer
```

---

## extract\_surface

```{eval-rst}
.. autofunction:: medunda.actions.extract_surface.extract_surface
```

---

## integrate\_between\_layers

```{eval-rst}
.. autofunction:: medunda.actions.integrate_between_layers.integrate_between_layers
```
