from pydantic import BaseModel
import geopandas as gpd

class Domain(BaseModel):
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float
    minimum_depth: float
    maximum_depth: float


GSA9 = Domain(
    minimum_latitude=41.29999921500007,
    maximum_latitude=44.42720294000003,
    minimum_longitude=7.525000098000021,
    maximum_longitude=13.003545339000027,
    minimum_depth=0,
    maximum_depth=800,
)

gsa_files = {"gsa_nine" : "C:\\Users\\akkar\\Desktop\\get_coords\\gsa_nine\\gsa_nine.shp",
            "adriatic" : "C:\\Users\\akkar\\Desktop\\get_coords\\adriatic_sea\\adriatic.shp",
            "gsa_med" : "C:\\Users\\akkar\\Desktop\\get_coords\\gsa_med\\GSAs_simplified_division.shp",
} 

def read_domain(shapefile_path, maximum_depth=800) -> Domain: 
    
    gdf = gpd.read_file(shapefile_path)
    
    xmin, ymin, xmax, ymax = gdf.total_bounds

    print(f"Longitude minimale: {xmin}")
    print(f"Longitude maximale: {xmax}")
    print(f"Latitude minimale: {ymin}")
    print(f"Latitude maximale: {ymax}")
    
    return Domain(minimum_latitude=ymin, maximum_latitude= ymax, minimum_longitude= xmin, maximum_longitude= xmax, minimum_depth=0, maximum_depth=maximum_depth)