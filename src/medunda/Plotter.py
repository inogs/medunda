
def extract_and_plot_layers(filepath, variable):
    """Extracts and plots surface, bottom, and average layers of the given variable."""
    
    ds = xr.open_dataset(filepath)
    
    surface = ds[variable].isel(depth=0)      # Surface layer (depth=0)
    bottom = ds[variable].isel(depth=-1)       # Bottom layer (last depth index)
    mean_layer = ds[variable].mean(dim="depth", skipna=True)      # Mean over the water column

    if not {'depth', 'latitude', 'longitude'}.issubset(ds[variable].dims) and not {'depth', 'lat', 'lon'}.issubset(data.dims):
        raise ValueError("The variable does not have the expected spatial dimensions (depth, lat/lon).")

    data=ds[variable]
    data=data.rename({'lat': 'latitude', 'lon':'longitude'}) if 'lat' in data else data
   
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    surface.plot(ax=axes[0], cmap="coolwarm")
    axes[0].set_title("Surface Layer")

    bottom.plot(ax=axes[1], cmap="coolwarm")
    axes[1].set_title("Bottom Layer")

    mean_layer.plot(ax=axes[2], cmap="coolwarm")
    axes[2].set_title("Mean over Water Column")

    plt.tight_layout()
    plt.show()

    ds.close()