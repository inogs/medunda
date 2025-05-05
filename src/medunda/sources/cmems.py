products = {     
       # -> each key represents a product and its associated with a list of available variables.
     # physical variables (MEDSEA_MULTIYEAR_PHY_006_004)
    "med-cmcc-tem-rean-m": ["thetao"],  # temperature
    "med-cmcc-cur-rean-m": ["uo"],    # current: composante zonale
    "med-cmcc-cur-rean-m": [ "vo" ],    # current: composante méridienne
    "med-cmcc-sal-rean-m" : ["so"],  # salinity

    # biogeochemical variables (MEDSEA_MULTIYEAR_BGC_006_008)
    "med-ogs-bio-rean-m": ["o2"],            # dissolved oxygen
    "med-ogs-car-rean-m":  ["ph"],          # pH
    "med-ogs-nut-rean-m" : ["no3"],          # nitrate
    "med-ogs-nut-rean-m" : ["po4"],          # phosphate
    "med-ogs-nut-rean-m": ["si"],      # silicate
    "med-ogs-pft-rean-m"  : ["chl"] ,         # chlorophylle a
}