# PRODUCTS = {     
#        # -> each key represents a product and its associated with a list of available variables.
#      # physical variables (MEDSEA_MULTIYEAR_PHY_006_004)
#     "med-cmcc-tem-rean-m": ["thetao"],       # temperature
#     "med-cmcc-cur-rean-m": ["vo","uo"],      # currents: northward and eastward
#     "med-cmcc-sal-rean-m": ["so"],           # salinity

#     # biogeochemical variables (MEDSEA_MULTIYEAR_BGC_006_008)
#     "med-ogs-bio-rean-m": ["o2"],                # dissolved oxygen
#     "med-ogs-car-rean-m": ["ph"],               # pH
#     "med-ogs-nut-rean-m": ["no3","po4","si"],   # nutrients: nitrate, phosphate and silicate
#     "med-ogs-pft-rean-m": ["chl"] ,            # chlorophylle a
# }

PRODUCTS={
    ("thetao",): ["med-cmcc-tem-rean-d", "med-cmcc-tem-rean-m"],
    ("vo", "uo"): ["med-cmcc-cur-rean-d", "med-cmcc-cur-rean-m"],
    ("so",): ["med-cmcc-cur-rean-d", "med-cmcc-sal-rean-m"],

    ("pH",): ["med-ogs-car-rean-d", "med-ogs-car-rean-m"],
    ("no3","po4","si"): ["med-ogs-nut-rean-d", "med-ogs-nut-rean-m"],
    ("chl",): ["med-ogs-pft-rean-d", "med-ogs-pft-rean-m"],
    ("o2",): ["med-ogs-bio-rean-d", "med-ogs-bio-rean-m"]
}

VARIABLES = []
for _var_list in PRODUCTS.keys():
    VARIABLES.extend(_var_list)
VARIABLES.sort()


def search_for_product(var_name: str, frequency:str) -> str:
    """ Given the name of a variable and a frequency, return the name of the CMEMS product that
    contains such variable with the specified frequency."""

    frequency_index = {"daily":0, "monthly":1}
    if frequency not in frequency_index:
        raise ValueError ("invalid frequency; use 'daily' or 'monthly'.")
    freq_index = frequency_index[frequency]

    selected_product = None
    for vars_tuple, prod_list in PRODUCTS.items():    
        if var_name in vars_tuple:
            selected_product = prod_list[freq_index]
            break

    if selected_product is None:
        raise ValueError (f"Variable '{var_name}' is not available in the dictionary")
    
    print(f"DEBUG: var_name={var_name}, selected_product={selected_product}")

    return selected_product
