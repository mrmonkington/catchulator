from dataclasses import dataclass
import csv
from geopandas.tools import geocode
from geopandas import read_file, GeoSeries, GeoDataFrame, sjoin
import yaml
import click
import re

SORT_KEY_MIN=-10e9
SORT_KEY_MAX=10e9

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='example.log', encoding='utf-8', level=logging.DEBUG)

def d(msg):
    logging.debug(msg)


@dataclass
class School:
    slug: str
    name: str
    # published admissions number
    pan: int
    total_places: int
    offers_made: int
    offers_accepted: int
    first_prefs_received: int
    first_prefs_accepted: float
    second_prefs_received: int
    second_prefs_accepted: float
    third_prefs_received: int
    third_prefs_accepted: float
    fsm_eligible_percent: float
    apps_expected: int = 0
    fsm_expected: int = 0

@dataclass
class Application:
    school: School
    sort_key: int = 0 
    def lottery(self):
        self.sort_key = random.randint(SORT_KEY_MIN, SORT_KEY_MAX)


@dataclass
class Child:
    pref_1: School
    pref_2: School
    pref_3: School

@dataclass
class Catchment:
    slug: str
    name: str
    # total of all schools in catchment
    pan: int = 0
    # how many children live in catchment
    apps_actual: int = 0
    # how many children actually qualify for FSM in this area 
    fsm_actual: int = 0
    # geometry

@dataclass
class Option:
    slug: str
    name: str
    catchments: dict[Catchment]
    geo: GeoDataFrame 


@dataclass
class LA:
    name: str
    pan: int # total of all schools in LA
    apps_expected: int
    fsm_expected: int


def read_school_data(filename):
    schools = {}
    with open(filename, newline='') as file_obj:
        reader_object = csv.DictReader(file_obj)
        for row in reader_object:
            schools[row['school_id']] = School(
                slug=row['school_id'],
                name=row['school_name'],
                pan=int(row['pan_2024']),
                total_places=int(row['total_number_places_offered']),
                offers_made=int(row['number_preferred_offers']),
                offers_accepted=int(row['number_1st_preference_offers']),
                first_prefs_received=int(row['times_put_as_1st_preference']),
                first_prefs_accepted=float(row['proportion_1stprefs_v_1stprefoffers']),
                second_prefs_received=int(row['times_put_as_2nd_preference']),
                second_prefs_accepted=float(row['proportion_1stprefs_v_totaloffers']),
                third_prefs_received=int(row['times_put_as_3rd_preference']),
                third_prefs_accepted=float(row['proportion_1stprefs_v_1stprefoffers']),
                fsm_eligible_percent=float(row['FSM_eligible_percent'])
            )
    return schools

def geocode_postcode(postcode) -> GeoDataFrame:
    postcode_locations = geocode("BN1 1AA, UK", provider="Nominatim", domain="localhost:8080", scheme="http")
    location = postcode_locations.get_geometry(0)
    location_gdf = GeoDataFrame({'geometry': location}, crs="EPSG:4326")
    return location_gdf

def load_options() -> dict[str,Catchment]:
    options = {}
    with open("catchments/options.yaml", newline='') as file_obj:
        options_yml: list = yaml.safe_load(file_obj)
        for opt_yml in options_yml:
            catchments: dict[str,Catchment] = {}
            for cmt_yml in opt_yml["catchments"]:
                catchments[cmt_yml["slug"]] = Catchment(
                    **cmt_yml
                )
            options[opt_yml["slug"]] = Option( 
                slug = opt_yml["slug"],
                name = opt_yml["name"],
                catchments = catchments,
                geo = read_file(f"catchments/{opt_yml['geofile']}")
            )
    return options

def find_catchment(postcode, option):
    location_gdf = geocode_postcode(postcode)
    joined_gdf = sjoin(option.geo, location_gdf, how='inner', predicate='intersects')

    if not joined_gdf.empty:
        overlapping_feature = joined_gdf.iloc[0]
        d(f"found catchment {overlapping_feature['catchment']} for postcode {postcode}")
        return overlapping_feature['catchment']
    else:
        d("Point doesn't intersect with any polygon")
        return False

def validate_postcode(ctx, param, value):
    # makes sure address has a postcode in it

    if re.match('[a-z]{1,2}\d[a-z\d]?\s*\d[a-z]{2}', value, flags=re.IGNORECASE):
        return value
    else:
        return click.BadParameter('Not a valid postcode!')


schools_data: dict = read_school_data('secondary_admissions_actuals_2425.csv')
options: dict = load_options()

#for school_id, school in schools_data.items():
#    d((school.name, school.pan, school.total_places, school.offers_made, school.offers_accepted))
#    #print(school)

#python catchsim.py --postcode "BN1 1AA" --option [A,B,C] --prefs varndean stringer patcham
@click.command()
@click.option('--postcode', required=True, type=str, callback=validate_postcode)
@click.option('--option', required=True, type=click.Choice(['A', 'B', 'C'], case_sensitive=False))
@click.option('--prefs', required=True, nargs=3, type=str)
def cli(postcode, option, prefs):
    print(find_catchment(postcode, options[option]))
    

if __name__ == "__main__":
    cli()
