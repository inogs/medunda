import logging
from enum import Enum


LOGGER = logging.getLogger(__name__)

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


class Frequency(Enum):
    DAILY = 0
    MONTHLY = 1



PRODUCTS={
    ("thetao",): {Frequency.DAILY: "med-cmcc-tem-rean-d", Frequency.MONTHLY: "med-cmcc-tem-rean-m"},
    ("vo", "uo"): {Frequency.DAILY: "med-cmcc-cur-rean-d", Frequency.MONTHLY:"med-cmcc-cur-rean-m"},
    ("so",): {Frequency.DAILY: "med-cmcc-cur-rean-d", Frequency.MONTHLY: "med-cmcc-sal-rean-m"},

    ("ph",): {Frequency.DAILY: "med-ogs-car-rean-d", Frequency.MONTHLY: "med-ogs-car-rean-m"},
    ("no3","po4","si"): {Frequency.DAILY: "med-ogs-nut-rean-d", Frequency.MONTHLY: "med-ogs-nut-rean-m"},
    ("chl",): {Frequency.DAILY: "med-ogs-pft-rean-d", Frequency.MONTHLY: "med-ogs-pft-rean-m"},
    ("o2",): {Frequency.DAILY: "med-ogs-bio-rean-d", Frequency.MONTHLY: "med-ogs-bio-rean-m"}
}

VARIABLES = []
for _var_list in PRODUCTS.keys():
    VARIABLES.extend(_var_list)
VARIABLES.sort()


def search_for_product(var_name: str, frequency:str) -> str:
    """ Given the name of a variable and a frequency, return the name of the CMEMS product that
    contains such variable with the specified frequency."""

    frequency_map = {"daily": Frequency.DAILY, "monthly": Frequency.MONTHLY}
    if frequency not in frequency_map:
        raise ValueError ("invalid frequency; use 'daily' or 'monthly'.")
    frequency_index = frequency_map[frequency]

    selected_product = None
    for vars_tuple, prod_dict in PRODUCTS.items():    
        if var_name in vars_tuple:
            selected_product = prod_dict[frequency_index]
            break

    if selected_product is None:
        raise ValueError (f"Variable '{var_name}' is not available in the dictionary")
    
    LOGGER.debug(f"var_name={var_name}, selected_product={selected_product}")

    return selected_product
