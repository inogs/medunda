import logging
from enum import Enum


LOGGER = logging.getLogger(__name__)



class Frequency(Enum):
    DAILY = 0
    MONTHLY = 1



PRODUCTS={
    #"MEDSEA_MULTIYEAR_PHY_006_004": 
    ("thetao",): 
        {Frequency.DAILY: "med-cmcc-tem-rean-d", 
        Frequency.MONTHLY: "med-cmcc-tem-rean-m"},
    ("vo", "uo"): 
        {Frequency.DAILY: "med-cmcc-cur-rean-d", 
        Frequency.MONTHLY:"med-cmcc-cur-rean-m"},
    ("so",): 
        {Frequency.DAILY: "med-cmcc-sal-rean-d", 
        Frequency.MONTHLY: "med-cmcc-sal-rean-m"},

    #"MEDSEA_MULTIYEAR_BGC_006_008":
    ("ph",):
         {Frequency.DAILY: "med-ogs-car-rean-d", 
         Frequency.MONTHLY: "med-ogs-car-rean-m"},
    ("no3","po4","si"): 
        {Frequency.DAILY: "med-ogs-nut-rean-d", 
         Frequency.MONTHLY: "med-ogs-nut-rean-m"},
    ("chl",): 
        {Frequency.DAILY: "med-ogs-pft-rean-d", 
         Frequency.MONTHLY: "med-ogs-pft-rean-m"},
    ("o2",): 
        {Frequency.DAILY: "med-ogs-bio-rean-d", 
         Frequency.MONTHLY: "med-ogs-bio-rean-m"},
    ("nppv",): 
         {Frequency.DAILY: "med-ogs-bio-rean-d", 
         Frequency.MONTHLY: "med-ogs-bio-rean-m"},
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
