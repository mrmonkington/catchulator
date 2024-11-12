from dataclasses import dataclass, field
import csv
from geopandas.tools import geocode
from geopandas import read_file, GeoSeries, GeoDataFrame, sjoin
import yaml
import click
import re
import pandas as pd
import copy
from pprint import pprint, pformat
import weakref
import random

from itertools import count
import sys
import math

# use a population scaling for CASA's projections to create same populations as estimated
# by BHCC
POP_SCALING = {
    "2024": 2267.0/2560.0,
    "2026": 2276.0/2410.0,
    "2030": 2023.0/2214.0
}

def stack_size2a(size=2):
    """Get stack size for caller's frame.
    """
    frame = sys._getframe(size)

    for size in count(size):
        frame = frame.f_back
        if not frame:
            return size


SORT_KEY_MIN=int(-10e9)
SORT_KEY_MAX=int(10e9)

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

lsoa_gdf = read_file('BrightonLSOA_Clean.geojson')

def d(msg):
    s = stack_size2a()
    logging.debug((s*" ") + str(msg))
def dtab(d: dict):
    logging.debug("\n" + pformat(d))

def o(msg):
    print(str(msg))
def otab(d: dict):
    print(pformat(d))

def get_lsoa_centroid(lsoa):
    return GeoDataFrame(
        { 'geometry': lsoa_gdf.to_crs('+proj=cea')[lsoa_gdf['lsoa21cd'] == \
            lsoa].centroid.to_crs('EPSG:4326') },
        crs="EPSG:4326"
    )


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
class Child:
    lsoa: str
    # 0 = 1st pref, etc
    prefs: list = None
    geo: GeoDataFrame = None
    catchment: Catchment = None
    uid: int = None
    def locate(self, option):
        # some special cases for when LSOA centroid falls out of the catchment maps
        if self.lsoa == "E01016999":
            self.geo = geocode_postcode("BN1 5LP")
        elif self.lsoa == "E01016916":
            self.geo = geocode_postcode("BN41 2WP")
        elif self.lsoa == "E01016885":
            self.geo = geocode_postcode("BN3 8GP")
        elif self.lsoa == "E01017011":
            self.geo = geocode_postcode("BN2 6SG")
        else:
            self.geo = get_lsoa_centroid(self.lsoa)
        self.catchment = option.catchments[reverse_catchment(self.geo, option.geo)]
    def __hash__(self):
        return self.id



@dataclass
class School:
    slug: str
    name: str
    urn: str
    geo_scaling: float
    popularity_scaling: float
    oversubscription_penalty: float
    # published admissions number
    pan: int
    total_places: int
    offers_made: int
    offers_accepted: int
    first_prefs_received: int
    #first_prefs_accepted: float
    second_prefs_received: int
    #second_prefs_accepted: float
    third_prefs_received: int
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
        self.remaining = self.pan

    def place(self, applications: list):
        c = len(applications)
        d(f"placing {c} in {self.slug} (remaining {self.remaining})")
        # d(applications)
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
    for child in children:
        for pref, school in enumerate(child.prefs):
            app = Application(
                child=child,
                school=school,
                pref=pref
            )
            app.lottery()
            applications.append(app)
    return applications

def summarise_app_children(option, schools, applications):
    o("Children in catchment summary")
    ctot = {}
    for cs, ca in option.catchments.items():
        ctot[cs] = 0
    children = set()
    for app in applications:
        children.add((app.child.uid, app.child.catchment.slug))
    for child, cs in children:
        ctot[cs] += 1
    otab(ctot)

def summarise_applications(option, schools, applications):
    o("Applications summary")
    schools_totals = {}
    for school in schools:
        schools_totals[school] = list((0,0,0,0,0,0))
    totals = list((0,0,0,0,0,0))
    for app in applications:
        schools_totals[app.school.slug][app.pref] += 1
        totals[app.pref] += 1
    o("Schools Totals")
    otab(schools_totals)
    o("Totals")
    o(totals)
    summarise_app_children(option, schools, applications)
    #d(applications)

def summarise_placed(schools):
    o("Placed summary")
    t = 0
    for school_id, school in schools.items():
        o(f"Total placed for {school_id} is {len(school.placed)}")
        t += len(school.placed)
    o(f" - Placed {t} in total")
            
def find_qualified_apps(school, applications) -> list[Application]:
    d(f"Finding applicants for {school.slug}")
    qualified=list(filter(lambda s: s.school.slug == school.slug, applications))
    qualified.sort(key=lambda a: a.sort_key)
    c = len(qualified)
    if c > 0:
        d(qualified[0].sort_key)
    d(f" - Found {c} qualified apps")
    # return qualified[0:count]
    return qualified

def search_applications(needle, haystack):
    for comp in haystack:
        if comp is needle:
            return True
    return False

def accept_offers(applications, placed, maximum, preference) -> list[Application]:
    d(f"accepting offers if pref {preference}")
    # reduces the applications list according to whether preference was first or not
    placed = list(app for app in placed if app.pref == preference)
    placed = placed[0:maximum]
    # get the kids placed
    children_placed = list(app.child for app in placed)
    # remove all applications from placed children
    applications_remaining = list(app for app in applications if not search_applications(app.child, children_placed))
    ca = len(applications_remaining)
    cp = len(placed)
    d(f"accepting {cp} offers leaving {ca} apps")
    return placed, applications_remaining

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

def find_qualified_apps_in_catchment(school, applications) -> float:
    d(f"Find apps in catchment {school.catchment.slug} for {school.slug}")
    # qualified = list()
    # for s in applications:
    qualified=list(
        filter(lambda s: s.school.slug == school.slug and school.catchment.slug == s.child.catchment.slug, applications)
    )
    qualified.sort(key=lambda a: a.sort_key)
    c = len(qualified)
    d(f" - Found {c} qualified apps")
    return qualified
    # return qualified[0:school.remaining]

def calculate_catchment_chance(applications, school) -> float:
    # include self
    expected = len(applications) + 1
    # can't place more FSM than we have places remaining after EHCP, sibling, etc
    remaining = school.remaining
    if remaining == 0:
        return 0.0
    if expected <= remaining:
        return 1.0
    return float(remaining)/float(expected)


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

def create_population(schools, option, year) -> list:
    # read in flows for LSOAs and populations, and synthesise
    # a list of imaginary children with school preferences
    # based on distance alone
    population_centres: dict = {}
    with open("bh_lsoa_projections_2020_31.csv") as file_obj:
        reader_object = csv.DictReader(file_obj)
        for row in reader_object:
            # adjust the CASA projections to match BHCC projections
            population_centres[row["geography"]] = round(
                float(row[f"{year}_scaled_BTN_places"]) * POP_SCALING[year]
            )
            #population_centres[row["geography"]] = round(float(row[f"{year}_realistic_apps"]))

    # read the flows
    # orig,dest,mins,flow
    #           ^    ^- actually the flow 
    #           +--- distance from child to school
    flows = pd.read_csv("simple_ttm3.csv", index_col=["orig", "dest"], usecols=["orig", "dest", "mins"])

    children = list()
    for lsoa, pop in population_centres.items():
        pop_flow = list()
        template_child: Child = Child(
            lsoa=lsoa
        )
        d(f"LSOA {lsoa} ")
        template_child.locate(option)
        for school_id, school in schools.items():
            urn = school.urn
            # how hard to get to school? Start with 'mins away'
            # higher is worse
            mins = flows.loc[(lsoa, urn)].iloc[0]
            # make it a bit nonlinear using geo factors - e.g. faith schools attract from further away
            # higher is better
            difficulty = math.exp(-1 * school.geo_scaling * mins)
            # scale by popularity factors - e.g. faith schools don't attract atheists
            # e.g. poor ofsted will count against a little
            # higher is better
            difficulty = difficulty * school.popularity_scaling
            # also consider in catchment and oversubscription - no point going for a popular school out of catchment
            # higher is better
            if template_child.catchment.slug != school.catchment.slug:
               difficulty = difficulty * school.oversubscription_penalty
            pop_flow.append((urn, difficulty))
        # d(pop_flow)
        # closest first
        pop_flow.sort(reverse=True, key=lambda s: s[1])
        # we use top 3 for actual prefs and remainder for fallback
        prefs = pop_flow
        # let's take 6 for now
        pref_template = [
            get_school_by_urn(schools, prefs[0][0]),
            get_school_by_urn(schools, prefs[1][0]),
            get_school_by_urn(schools, prefs[2][0]),
            get_school_by_urn(schools, prefs[3][0]),
            get_school_by_urn(schools, prefs[4][0]),
            get_school_by_urn(schools, prefs[5][0]),
        ]
        template_child.prefs=pref_template
        d(f"Prefs for {lsoa} x {pop} kids in catchment {template_child.catchment.slug}")
        d(pformat([p.slug for p in pref_template]))
        for i in range(0, pop):
            child_instance = copy.deepcopy(template_child)
            # just need an ID for hashing
            child_instance.uid = random.randint(SORT_KEY_MIN, SORT_KEY_MAX)
            children.append(child_instance)
        # d(f"Created {pop} children in {lsoa}")
    c = len(children)
    d(f"Created {c} children in total")
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
                geo_scaling=float(row['geo_scaling']),
                popularity_scaling=float(row['popularity_scaling']),
                oversubscription_penalty=float(row['oversubscription_penalty']),
                pan=int(row[f"pan_{year}"]),
                total_places=int(row['total_number_places_offered']),
                offers_made=int(row['number_preferred_offers']),
                offers_accepted=int(row['number_1st_preference_offers']),
                first_prefs_received=int(row['times_put_as_1st_preference']),
                #first_prefs_accepted=float(row['proportion_1stprefs_v_1stprefoffers']),
                second_prefs_received=int(row['times_put_as_2nd_preference']),
                #second_prefs_accepted=float(row['proportion_1stprefs_v_totaloffers']),
                third_prefs_received=int(row['times_put_as_3rd_preference']),
                #third_prefs_accepted=float(row['proportion_1stprefs_v_1stprefoffers']),
                fsm_expected=round(float(row['FSM_expected'])),
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

@click.group()
def main():
    pass

#python catchsim.py --postcode "BN1 1AA" --option [A,B,C] --prefs varndean stringer patcham
@main.command()
@click.option('--postcode', required=True, type=str, callback=validate_postcode)
@click.option('--option', required=True, type=click.Choice(['A', 'B', 'C', 'X'], case_sensitive=False))
@click.option('--year', required=True, type=click.Choice(['2024', '2026']))
@click.option('--prefs', required=True, nargs=3, type=str)
@click.option('--debug', is_flag=True, default=False)
def sim(postcode, option, year, prefs, debug):
    if(debug):
        logging.basicConfig(level=logging.WARN)

    schools_data: dict = read_school_data('secondary_admissions_actuals_2425.csv', options[option], year)

    user_catchment = find_catchment(postcode, options[option])
    d(f"Your catchment is {user_catchment}")
    children = create_population(schools_data, options[option], year)

    # look up preferred schools
    pref_school = []
    for pref in prefs:
        pref_school.append({
            'school': schools_data[pref],
            'catchment_chance': 0.0,
            'outofcatchment_chance': 0.0,
            'fallback_chance': 0.0,
            'fsm_chance': 0.0
        })

    # create big list of application objects, with random sort keys
    applications=create_applications(schools_data, children)

    summarise_applications(options[option], schools_data, applications)

    # assume some priotities get their first choice
    d(f"------- PLACING EHCP (FIRST PREF) -----------")
    for school_id, school in schools_data.items():
        # take some fixed % of population randomly as ehcp
        placed = find_qualified_apps(school, applications)
        placed, applications = accept_offers(applications, placed,
                                             min(school.remaining, school.ehcp_expected),
                                             preference=0)
        school.place(placed)

    d("\n")

    summarise_placed(schools_data)

    d(f"------- PLACING SIBLING (FIRST PREF) -----------")
    for school_id, school in schools_data.items():
        # take some fixed % of population randomly as siblings
        # faith schools are assumed to fill their entire intake in this
        # way as we have no insight into their admissions
        # TODO spread a bit further
        placed = find_qualified_apps(school, applications)
        placed, applications = accept_offers(applications, placed,
                                             min(school.remaining, school.sibling_expected),
                                             preference=0)
        school.place(placed)
    d("\n")

    summarise_placed(schools_data)

    # this would just be FSM target / FSM expected
    pref_school[0]["fsm_chance"] = calculate_fsm_chance(
        pref_school[0]["school"]
    )

    d(f"------- PLACING FSM (FIRST PREF) -----------")
    # assume FSM is hit on first pass so just do this once
    for school_id, school in schools_data.items():
        # take some fixed % of population randomly as fsm
        placed = find_qualified_apps(school, applications)
        placed, applications = accept_offers(
            applications, placed, min(school.fsm_target, school.remaining), preference=0
        )
        school.place(placed)
    d("\n")

    summarise_applications(options[option], schools_data, applications)
    summarise_placed(schools_data)

    d(f"------- PLACING CATCHMENT -----------")
    for pref_rank in (0, 1, 2):
        d(f"------- PREF {pref_rank} -----------")
        for school_id, school in schools_data.items():
            placed = find_qualified_apps_in_catchment(school, applications)
            if school == pref_school:
                pref_school[pref_rank]["catchment_chance"] = calculate_catchment_chance(
                    placed, preference=pref_school[pref_rank]["school"]
                )
            placed, applications = accept_offers(applications, placed, school.remaining, preference=pref_rank)
            school.place(placed)
    d("\n")

    summarise_applications(options[option], schools_data, applications)
    summarise_placed(schools_data)

    d(f"------- PLACING OUT OF CATCHMENT -----------")
    for pref_rank in (0, 1, 2):
        d(f"------- PREF {pref_rank} -----------")
        for school_id, school in schools_data.items():
            placed = find_qualified_apps(school, applications)
            if school == pref_school:
                pref_school[pref_rank]["outofcatchment_chance"] = calculate_catchment_chance(
                    placed, preference=pref_school[pref_rank]["school"]
                )
            placed, applications = accept_offers(applications, placed, school.remaining, preference=pref_rank)
            school.place(placed)
    d("\n")

    summarise_applications(options[option], schools_data, applications)
    summarise_placed(schools_data)

    d(f"------- PLACING FALLBACK -----------")
    for pref_rank in (3, 4, 5):
        d(f"------- PREF {pref_rank} -----------")
        for school_id, school in schools_data.items():
            placed = find_qualified_apps(school, applications)
            if school == pref_school:
                pref_school[pref_rank]["fallback_chance"] = calculate_catchment_chance(
                    placed, preference=pref_school[pref_rank]["school"]
                )
            placed, applications = accept_offers(applications, placed, school.remaining, preference=pref_rank)
            school.place(placed)
    d("\n")

    d(f"------- SUMMARIES -----------")
    summarise_applications(options[option], schools_data, applications)
    summarise_placed(schools_data)

@main.command()
@click.option('--option', required=True, type=click.Choice(['A', 'B', 'C', 'X'], case_sensitive=False))
@click.option('--year', required=True, type=click.Choice(['2024', '2026']))
def anneal(option, year):
    # used for simulated annleaing of preference factors
    # usage:
    # pip install jupyterlab
    # jupyter lab
    # > improt catchsim
    # > mini = catchsim.anneal("B", 2024)
    # wait an hour and get final params
    # > mini.current_state
    option = options[option]

    schools_data: dict = read_school_data('secondary_admissions_actuals_2425.csv', option, year)

    # vars to iterate
    # geo_scaling
    # popularity_scaling
    geo_scaling = [school.geo_scaling for school_slug, school in schools_data.items()]
    popularity_scaling = [school.popularity_scaling for school_slug, school in schools_data.items()]
    from anneal import minimize
    from functools import partial
    x0 = geo_scaling + popularity_scaling
    cost_func = partial(anneal_iterate, schools_data, option, year)
    # min max
    bounds = ((0.0001, 30.0),)*len(geo_scaling)*2
    mini = minimize(cost_func, x0, opt_mode='continuous', t_max=500,
                    bounds=bounds, cooling_schedule='linear', damping=0.001)
    mini.results()
    o("Final params")
    ns = len(schools_data)
    o("print geo:")
    o(mini.current_state[0:ns])
    o("print pop:")
    o(mini.current_state[ns:])


def get_prefs_dist(schools, children):
    schools_totals = {}
    for school in schools:
        schools_totals[school] = list((0,0,0,0,0,0))
    for child in children:
        for pref_rank in (0,1,2):
            schools_totals[child.prefs[pref_rank].slug][pref_rank] += 1
    return schools_totals

def anneal_cost(schools, applications):
    dist = get_prefs_dist(schools, applications)
    print(dist)
    cost = 0
    for school_slug, school in schools.items():
        cost += abs(dist[school_slug][0] - school.first_prefs_received)
        cost += abs(dist[school_slug][1] - school.second_prefs_received)
        cost += abs(dist[school_slug][2] - school.third_prefs_received)
    return cost

def anneal_iterate(schools_data, option, year, x):
    # merge schools_data and scalings
    ns = len(schools_data)
    for c, (school_slug, school) in enumerate(schools_data.items()):
        school.geo_scaling = x[c]
        school.popularity_scaling = x[ns+c]
    children = create_population(schools_data, option, year)
    cost = anneal_cost(schools_data, children)
    return cost


if __name__ == "__main__":
    main()

