from geopandas.tools import geocode
from geopandas import read_file, GeoSeries, GeoDataFrame, sjoin

postcode_locations = geocode("BN1 1AA, UK", provider="Nominatim", domain="localhost:8080", scheme="http")
location = postcode_locations.get_geometry(0)


cmt = read_file("catchments/catchment_03.geojson")
print(cmt)
location_gdf = GeoDataFrame({'geometry': location}, crs=cmt.crs)
joined_gdf = sjoin(cmt, location_gdf, how='inner', predicate='intersects')

if not joined_gdf.empty:
    overlapping_feature = joined_gdf.iloc[0]
    print(overlapping_feature['catchment'])  # Replace 'attribute_name' with the desired attribute
else:
    print("Point doesn't intersect with any polygon")
#cmt.sindex
