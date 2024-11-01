from dataclasses import dataclass, field
import csv
from geopandas.tools import geocode
from geopandas import read_file, GeoSeries, GeoDataFrame, sjoin
import yaml
import click
import re
import pandas as pd
import copy
from pprint import pprint
import weakref


SORT_KEY_MIN=-10e9
SORT_KEY_MAX=10e9

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)

lsoa_gdf = read_file('BrightonLSOA_Clean.geojson')

def d(msg):
    logging.debug(msg)

def get_lsoa_centroid(lsoa):
    return gpd.GeoDataFrame(
        { 'geometry': gdf[gdf['lsoa21cd'] == lsoa].centroid },
        crs="EPSG:4326"
    )


@dataclass
class Child:
    lsoa: str
    # 0 = 1st pref, etc
    prefs: list
    geo: GeoDataFrame = None
    def __post__init__(self):
        self.geo = get_lsoa_centroid(self.lsoa)
        self.catchment = reverse_catchment(self.geo, option.geo)


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
class School:
    slug: str
    name: str
    urn: str
    # published admissions number
    pan: int
    total_places: int
    offers_made: int
    offers_accepted: int
    #first_prefs_received: int
    #first_prefs_accepted: float
    #second_prefs_received: int
    #second_prefs_accepted: float
    #third_prefs_received: int
    #third_prefs_accepted: float
    #fsm_eligible_percent: float
    fsm_target: int
    fsm_expected: int
    sibling_expected: int
    ehcp_expected: int
    catchment: Catchment
    apps_expected: int = 0
    placed: list = field(default_factory=list)
    remaining: int = 0

    def __post_init__(self):
        self.remaining = self.total_places

    def place(self, applications: list):
        if len(applications) > self.remaining:
            raise("Tried to place too many applications")
        self.placed += applications
        self.remaining -= len(applications)


@dataclass
class Application:
    child: Child
    school: School
    # 0 = 1st pref, etc
    pref: int = 0
    sort_key: int = 0 
    def lottery(self):
        self.sort_key = random.randint(SORT_KEY_MIN, SORT_KEY_MAX)


def create_applications(schools, children) -> list[Application]: 
    applications = []
    for child in Children:
        for pref, school in enumerate(child.prefs):
            app = Application(
                chlid=child,
                school=school,
                pref=pref
            )
            app.lottery()
            applications.append(app)
    return applications
            
def fill_places(school, applications, count) -> list[Child]:
    qualified=filter(lambda s: s.school == school, applications)
    qualified.sort(lambda a: a.sort_key)
    return qualified[0:count]

def fill_ehcp(school, applications):
    return fill_places(school, applications, school.ehcp_expected)

def fill_sibling(school, applications):
    return fill_places(school, applications, school.sibling_expected)

def fill_fsm(school, applications):
    return fill_places(school, applications, school.fsm_target)

def accept_offers(applications, placed, preference) -> list[Application]:
    # reduces the applications list according to whether preference was first or not
    pass

def calculate_fsm_chance(school) -> float:
    # include self
    expected = school.fsm_expected + 1
    # can't place more FSM than we have places remaining after EHCP, sibling, etc
    remaining = min(school.remaining, school.fsm_target)
    if remaining == 0:
        return 0.0
    if expected <= remaining:
        return 1.0
    return float(remaining)/float(expected)

def calculate_catchment_chance(applications, school, preference) -> float:
    qualified=filter(lambda s: s.school == school, applications)
    qualified.sort(lambda a: a.sort_key)
    return qualified[0:count]


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

def create_population(schools) -> list:
    population_centres: dict = {}
    with open("bh_lsoa_projections_2020_31.csv") as file_obj:
        reader_object = csv.DictReader(file_obj)
        for row in reader_object:
            population_centres[row["geography"]] = round(float(row["2026_scaled_BTN_places"]))

    d("population centres:")
    d(population_centres)
    # orig,dest,flow,btn_cij_flow$flow
    #                ^- actually the flow 
    flows = pd.read_csv("simple_ttm3.csv", index_col=["orig", "dest"], usecols=["orig", "dest", "flow"])
    d("flows")
    d(flows)

    children = []
    for lsoa, pop in population_centres.items():
        pop_flow = []
        for school_id, school in schools.items():
            urn = school.urn
            pop_flow.append((urn,
                             flows.loc[(lsoa, urn)].iloc[0]))
        d(pop_flow)
        pop_flow.sort(reverse=True, key=lambda s: s[1])
        # get top 3
        prefs = pop_flow[0:3]
        d("Prefs for {(lsoa, urn)}")
        d(prefs)
        template_child: Child = Child(
            lsoa=lsoa,
            prefs=[
                get_school_by_urn(schools, prefs[0][0]).slug,
                get_school_by_urn(schools, prefs[1][0]).slug,
                get_school_by_urn(schools, prefs[2][0]).slug
            ]
        )
        for i in range(0, pop):
            children.append(copy.deepcopy(template_child))
    return children


def read_school_data(filename, option, year):
    # model is X,A,B,C
    # year is 2024 or 2026
    schools = {}
    with open(filename) as file_obj:
        reader_object = csv.DictReader(file_obj)
        for row in reader_object:
            schools[row['school_id']] = School(
                slug=row['school_id'],
                urn=int(row['school_urn']),
                name=row['school_name'],
                pan=int(row['pan_2026']),
                total_places=int(row['total_number_places_offered']),
                offers_made=int(row['number_preferred_offers']),
                offers_accepted=int(row['number_1st_preference_offers']),
                #first_prefs_received=int(row['times_put_as_1st_preference']),
                #first_prefs_accepted=float(row['proportion_1stprefs_v_1stprefoffers']),
                #second_prefs_received=int(row['times_put_as_2nd_preference']),
                #second_prefs_accepted=float(row['proportion_1stprefs_v_totaloffers']),
                #third_prefs_received=int(row['times_put_as_3rd_preference']),
                #third_prefs_accepted=float(row['proportion_1stprefs_v_1stprefoffers']),
                fsm_expected=float(row['FSM_expected']),
                fsm_target=round(float(row['FSM_target'])),
                ehcp_expected=round(float(row['p2_a_EHCP_off'])),
                sibling_expected=round(float(row['p3_sibling_offered'])),
                catchment=option.catchments[str(row[f"option{option.slug}"])],
            )
    return schools

def geocode_postcode(postcode) -> GeoDataFrame:
    postcode_locations = geocode(f"{postcode}, UK", provider="Nominatim", domain="localhost:8080", scheme="http")
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
    return reverse_catchment(location_gdf, option.geo)

def reverse_catchment(point_geo, catchment_geo):
    joined_gdf = sjoin(catchment_geo, point_geo, how='inner', predicate='intersects')

    if not joined_gdf.empty:
        overlapping_feature = joined_gdf.iloc[0]
        d(f"found catchment {overlapping_feature['catchment']}")
        return overlapping_feature['catchment']
    else:
        d("Point doesn't intersect with any polygon")
        return False

def validate_postcode(ctx, param, value):
    # makes sure address has a postcode in it

    if re.search(r'^[a-z]{1,2}\d[a-z\d]?\s*\d[a-z]{2}$', value, flags=re.IGNORECASE):
        return value
    else:
        raise click.BadParameter('Not a valid postcode!')

def get_school_by_urn(schools, urn):
    return next(s for sid, s in schools.items() if s.urn == urn)


options: dict = load_options()


#python catchsim.py --postcode "BN1 1AA" --option [A,B,C] --prefs varndean stringer patcham
@click.command()
@click.option('--postcode', required=True, type=str, callback=validate_postcode)
@click.option('--option', required=True, type=click.Choice(['A', 'B', 'C'], case_sensitive=False))
@click.option('--year', required=True, type=click.Choice(['2024', '2026']))
@click.option('--prefs', required=True, nargs=3, type=str)
@click.option('--debug', is_flag=True, default=False)
def cli(postcode, option, year, prefs, debug):
    if(debug):
        logging.basicConfig(level=logging.DEBUG)

    schools_data: dict = read_school_data('secondary_admissions_actuals_2425.csv', options[option], year)
    #for school_id, school in schools_data.items():
    #    d((school.urn, school.name, school.pan, school.total_places, school.offers_made, school.offers_accepted))
    #    #print(school)

    user_catchment = find_catchment(postcode, options[option])
    d("Your catchment is {user_catchment}")
    children = create_population(schools_data)
    pprint(children)

    # look up preferred schools
    pref_school = []
    for pref in prefs:
        pref_school.append({
            'school': schools_data[pref],
            'catchment_chance': 0.0,
            'fsm_chance': 0.0
        })

    # create big list of application objects, with random sort keys
    applications=create_applications(schools_data, children)

    for school_id, school in schools_data.items():
        # take some fixed % of population randomly as ehcp
        placed=fill_ehcp(school, applications)
        placed=accept_offers(applications, placed, preference=1)
        school.place(placed)

    for school_id, school in schools_data.items():
        # take some fixed % of population randomly as siblings
        # faith schools are assumed to fill their entire intake in this
        # way as we have no insight into their admissions
        # TODO spread a bit further
        placed=fill_sibling(school, applications)
        placed=accept_offers(applications, placed, preference=1)
        school.place(placed)

    # this would just be FSM target / FSM expected
    pref_school[0]["fsm_chance"] = calculate_fsm_chance(
        pref_school[0]["school"]
    )

    for school_id, school in schools_data.items():
        # take some fixed % of population randomly as fsm
        placed=fill_fsm(school, applications)
        placed=accept_offers(applications, placed, preference=1)
        school_id.place(placed)

    pref_school[0]["catchment_chance"] = calculate_chance(application, preference=pref_school[0]["school"])
    placed=fill_catchment(schools, applications)
    remove_applications(applications, placed, preference=1)


if __name__ == "__main__":
    cli()
