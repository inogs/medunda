PRODUCTS = {     
       # -> each key represents a product and its associated with a list of available variables.
     # physical variables (MEDSEA_MULTIYEAR_PHY_006_004)
    "med-cmcc-tem-rean-m": ["thetao"],       # temperature
    "med-cmcc-cur-rean-m": ["vo","uo"],      # currents: northward and eastward
    "med-cmcc-sal-rean-m": ["so"],           # salinity

    # biogeochemical variables (MEDSEA_MULTIYEAR_BGC_006_008)
    "med-ogs-bio-rean-m": ["o2"],                # dissolved oxygen
    "med-ogs-car-rean-m": ["ph"],               # pH
    "med-ogs-nut-rean-m": ["no3","po4","si"],   # nutrients: nitrate, phosphate and silicate
    "med-ogs-pft-rean-m": ["chl"] ,            # chlorophylle a
}

VARIABLES = []
for _var_list in PRODUCTS.values():
    VARIABLES.extend(_var_list)
VARIABLES.sort()


def search_for_product(var_name: str) -> str:
    """ Given the name of a variable, return the name of the CMEMS product that
    contains such variable. """

    selected_product = None
    for prod_id, vars_available in PRODUCTS.items():             #zip: to loop between more keys
        if var_name in vars_available:
            selected_product = prod_id
            break

    if selected_product is None:
        raise ValueError (f"Variable '{var_name}' is not available in the dictionary")
    return selected_product
