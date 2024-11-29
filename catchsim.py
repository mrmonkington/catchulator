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

import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

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


lsoa_gdf = read_file('BrightonLSOA_Clean.geojson')

def d(msg):
    s = stack_size2a()
    logger.debug((s*" ") + str(msg))
def dtab(d: dict):
    logger.debug("\n" + pformat(d))

def o(msg):
    s = stack_size2a()
    logger.info((s*" ") + str(msg))
def otab(d: dict):
    logger.info("\n" + pformat(d))

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
    out_of_catchment_pri: bool = False
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
    catchment_capacity: int = 0
    # out of catchment capacity
    oo_catchment_capacity: int = 0

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
        d(f" -> placed {c} in {self.slug} (remaining {self.remaining})")


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
    d("Children in catchment summary")
    ctot = {}
    for cs, ca in option.catchments.items():
        ctot[cs] = 0
    children = set()
    for app in applications:
        children.add((app.child.uid, app.child.catchment.slug))
    for child, cs in children:
        ctot[cs] += 1
    dtab(ctot)

def summarise_applications(option, schools, applications):
    d("Applications summary")
    schools_totals = {}
    for school in schools:
        schools_totals[school] = list((0,0,0,0,0,0))
    totals = list((0,0,0,0,0,0))
    for app in applications:
        schools_totals[app.school.slug][app.pref] += 1
        totals[app.pref] += 1
    d("Schools Totals")
    dtab(schools_totals)
    d("Totals")
    d(totals)
    summarise_app_children(option, schools, applications)
    #d(applications)

def summarise_preferences(children):
    d("preference summary")
    from collections import defaultdict
    prefs = defaultdict(int)
    for child in children:
        p = tuple(pref.slug for pref in child.prefs)
        prefs[p] += 1
    prefs = sorted(prefs.items(), key=lambda a: a[1], reverse=True)
    otab(prefs)
    #d(applications)

def summarise_placed(schools):
    d("Placed summary")
    t = 0
    for school_id, school in schools.items():
        d(f"Total placed for {school_id} is {len(school.placed)}")
        t += len(school.placed)
    d(f" - Placed {t} in total")
            
def find_qualified_apps(school, applications) -> list[Application]:
    d(f"Finding applicants for {school.slug}")
    qualified=list(filter(lambda s: s.school.slug == school.slug and s.pref in (0,1,2), applications))
    qualified.sort(key=lambda a: a.sort_key)
    c = len(qualified)
    if c > 0:
        d(qualified[0].sort_key)
    # d(f" - Found {c} qualified apps")
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
        filter(lambda s: s.school.slug == school.slug \
            and school.catchment.slug == s.child.catchment.slug \
            and s.pref in (0,1,2), applications)
    )
    qualified.sort(key=lambda a: a.sort_key)
    c = len(qualified)
    # d(f" - Found {c} qualified apps")
    return qualified
    # return qualified[0:school.remaining]

def calculate_oo_catchment_chance(applications, school) -> float:
    # include self
    expected = len(applications) + 1
    # can't place more FSM than we have places remaining after EHCP, sibling, etc
    remaining = school.oo_catchment_capacity
    d(f"School: {school.slug}")
    d(f"Remaining: {remaining}")
    d(f"Expected: {expected}")
    if remaining == 0:
        return 0.0
    if expected <= remaining:
        return 1.0
    return float(remaining)/float(expected)

def calculate_catchment_chance(applications, school) -> float:
    # include self
    expected = len(applications) + 1
    # can't place more FSM than we have places remaining after EHCP, sibling, etc
    remaining = school.catchment_capacity
    d(f"School: {school.slug}")
    d(f"Remaining: {remaining}")
    d(f"Expected: {expected}")
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
    # number of preferences in this model
    prefs: int = 3


@dataclass
class LA:
    name: str
    pan: int # total of all schools in LA
    apps_expected: int
    fsm_expected: int

def calc_school_score_basic(option, school, child, distance):
    """
        v1 of distance rolloff using exp(w * -x) curve, which is too steep
    """
    # make it a bit nonlinear using geo factors - e.g. faith schools attract from further away
    # higher is better
    # max is 1 when there is no distance between child and school
    score = math.exp(-1 * school.geo_scaling * distance)
    # scale by popularity factors - e.g. faith schools don't attract atheists
    # e.g. poor ofsted will count against a little
    # higher is better
    score = score * school.popularity_scaling
    # also consider in catchment and oversubscription - no point going for a popular school out of catchment
    # higher is better
    if child.catchment.slug != school.catchment.slug:
       score = score * school.oversubscription_penalty
    return score

def calc_school_score_cubic(option, school, child, distance):
    """
        v1 of distance rolloff using cubic easing curve from 0 -> geo_scaling mins
        with max of 1 and floor value of 0.2 (i.e. never zeroes out the score)
    """
    # this is effectively max distance most children would travel
    window_size = school.geo_scaling
    floor = 0.2
    # make it a bit nonlinear using geo factors - e.g. faith schools attract from further away
    # higher is better
    # max is 1 when there is no distance between child and school
    # use cubice easing at either end of window
    if distance > window_size:
        score = floor
    elif distance >= 0.0 and distance < (window_size / 2):
        score = 1 \
            - 4 * math.pow(distance / window_size, 3) * (1 - floor) \
            + floor
    elif distance >= (window_size / 2) and distance <= window_size:
        score =  pow(-2 * (distance / window_size) + 2, 3) / 2 * (1 - floor) \
            + floor
    else:
        # should never get here
        score = 1.0
    # scale by popularity factors - e.g. faith schools don't attract atheists
    # e.g. poor ofsted will count against a little
    # higher is better
    score = score * school.popularity_scaling
    # also consider in catchment and oversubscription - no point going for a popular school out of catchment
    # higher is better
    if child.catchment.slug != school.catchment.slug:
       score = score * school.oversubscription_penalty
    return score

def create_population(schools, option, popyear) -> list:
    # read in flows for LSOAs and populations, and synthesise
    # a list of imaginary children with school preferences
    # based on distance/reputation/expectation scoring
    population_centres: dict = {}
    with open("bh_lsoa_projections_2020_31.csv") as file_obj:
        reader_object = csv.DictReader(file_obj)
        for row in reader_object:
            # adjust the CASA projections to match BHCC projections
            population_centres[row["geography"]] = round(
                float(row[f"{popyear}_scaled_BTN_places"]) * POP_SCALING[popyear]
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
        # d(f"LSOA {lsoa} ")
        template_child.locate(option)
        for school_id, school in schools.items():
            urn = school.urn
            # how hard to get to school? Start with 'mins away'
            # higher is worse
            mins = flows.loc[(lsoa, urn)].iloc[0]
            # score = calc_school_score_basic(option, school, template_child, mins)
            score = calc_school_score_cubic(option, school, template_child, mins)
            pop_flow.append((urn, score))
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
        # d(f"Prefs for {lsoa} x {pop} kids in catchment {template_child.catchment.slug}")
        # d(pformat([p.slug for p in pref_template]))
        for i in range(0, pop):
            child_instance = copy.deepcopy(template_child)
            # just need an ID for hashing
            child_instance.uid = random.randint(SORT_KEY_MIN, SORT_KEY_MAX)
            children.append(child_instance)
        # d(f"Created {pop} children in {lsoa}")
    c = len(children)
    d(f"Created {c} children in total")
    return children


def read_school_data(filename, option, popyear, panyear, geo_scaling=None, popularity_scaling=None, oversubscription_penalty=None):
    # model is X,A,B,C
    # year is 2024 or 2026
    # geo_scaling and pop_scaling are override parameter lists, or None if no override
    #  and values in datafile should be used
    schools = {}
    with open(filename) as file_obj:
        reader_object = csv.DictReader(file_obj)
        for ind, row in enumerate(reader_object):
            s = School(
                slug=row['school_id'],
                urn=int(row['school_urn']),
                name=row['school_name'],
                geo_scaling=float(row['geo_scaling']),
                popularity_scaling=float(row['popularity_scaling']),
                oversubscription_penalty=float(row['oversubscription_penalty']),
                pan=int(row[f"pan_{panyear}"]),
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
            # override scaling - useful for param development
            if geo_scaling:
                s.geo_scaling = geo_scaling[ind]
            if popularity_scaling:
                s.popularity_scaling = popularity_scaling[ind]
            if oversubscription_penalty:
                s.oversubscription_penalty = oversubscription_penalty[ind]
            schools[row['school_id']] = s
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
        # d(f"found catchment {overlapping_feature['catchment']}")
        return overlapping_feature['catchment']
    else:
        # d("Point doesn't intersect with any polygon")
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
@click.option('--popyear', required=True, type=click.Choice(['2024', '2026']))
@click.option('--panyear', required=True, type=click.Choice(['2024', '2026']))
@click.option('--prefs', required=True, nargs=3, type=str)
@click.option('--debug', is_flag=True, default=False)
@click.option('--fsm', is_flag=True, default=False)
@click.option('--geo_scaling', required=False, type=str, default=None)
@click.option('--popularity_scaling', required=False, type=str, default=None)
@click.option('--oversubscription_penalty', required=False, type=str, default=None)
def sim(postcode, option, popyear, panyear, prefs, debug, geo_scaling, popularity_scaling, oversubscription_penalty, fsm):
    sim_run(postcode, option, popyear, panyear, prefs, debug, geo_scaling, popularity_scaling, oversubscription_penalty, fsm)

def sim_run(postcode, option, popyear, panyear, prefs, debug, geo_scaling, popularity_scaling, oversubscription_penalty, fsm):
    if(debug):
        logging.basicConfig(level=logging.DEBUG)



    if geo_scaling:
        geo_scaling = [float(c.strip()) for c in geo_scaling.split(',')]
    if popularity_scaling:
        popularity_scaling = [float(c.strip()) for c in popularity_scaling.split(',')]
    if oversubscription_penalty:
        oversubscription_penalty = [float(c.strip()) for c in oversubscription_penalty.split(',')]
    schools_data: dict = read_school_data('secondary_admissions_actuals_2425.csv', options[option],
                                          popyear,
                                          panyear,
                                          geo_scaling=geo_scaling,
                                          popularity_scaling=popularity_scaling,
                                          oversubscription_penalty=oversubscription_penalty)

    user_catchment = find_catchment(postcode, options[option])
    d(f"Your catchment is {user_catchment}")
    children = create_population(schools_data, options[option], popyear)
    summarise_preferences(children)

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

    # summarise_placed(schools_data)

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

    # summarise_placed(schools_data)

    # this would just be FSM target / FSM expected
    for pref_rank in (0, 1, 2):
        pref_school[pref_rank]["fsm_chance"] = calculate_fsm_chance(
            pref_school[pref_rank]["school"]
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
    # summarise_placed(schools_data)

    # set catchment capacity as current remaining spaces
    for school_id, school in schools_data.items():
        school.catchment_capacity = school.remaining

    d(f"------- PLACING CATCHMENT -----------")
    for pref_rank in (0, 1, 2):
        d(f"------- PREF {pref_rank} -----------")
        school = pref_school[pref_rank]["school"]
        if school.catchment.slug == user_catchment:
            qual = find_qualified_apps_in_catchment(school, applications)
            pref_school[pref_rank]["catchment_chance"] = calculate_catchment_chance(
                qual, pref_school[pref_rank]["school"]
            )
    for pref_rank in (0, 1, 2):
        d(f"------- PREF {pref_rank} -----------")
        for school_id, school in schools_data.items():
            qual = find_qualified_apps_in_catchment(school, applications)
            placed, applications = accept_offers(applications, qual, school.remaining, preference=pref_rank)
            school.place(placed)
    d("\n")

    #summarise_applications(options[option], schools_data, applications)
    #summarise_placed(schools_data)

    # set out of catchment capacity as current remaining spaces
    for school_id, school in schools_data.items():
        school.oo_catchment_capacity = school.remaining

    d(f"------- PLACING OUT OF CATCHMENT -----------")
    for pref_rank in (0, 1, 2):
        d(f"------- PREF {pref_rank} -----------")
        school = pref_school[pref_rank]["school"]
        if school.catchment.slug != user_catchment:
            qual = find_qualified_apps(school, applications)
            pref_school[pref_rank]["outofcatchment_chance"] = calculate_oo_catchment_chance(
                qual, pref_school[pref_rank]["school"]
            )
    for pref_rank in (0, 1, 2):
        d(f"------- PREF {pref_rank} -----------")
        for school_id, school in schools_data.items():
            qual = find_qualified_apps(school, applications)
            placed, applications = accept_offers(applications, qual, school.remaining, preference=pref_rank)
            school.place(placed)
    d("\n")

    #summarise_applications(options[option], schools_data, applications)
    #summarise_placed(schools_data)

    #d(f"------- PLACING FALLBACK -----------")
    #for pref_rank in (3, 4, 5):
    #    d(f"------- PREF {pref_rank} -----------")
    #    for school_id, school in schools_data.items():
    #        placed = find_qualified_apps(school, applications)
    #        if school.slug == pref_school[pref_rank]["school"].slug:
    #            pref_school[pref_rank]["fallback_chance"] = calculate_catchment_chance(
    #                placed, pref_school[pref_rank]["school"]
    #            )
    #        placed, applications = accept_offers(applications, placed, school.remaining, preference=pref_rank)
    #        school.place(placed)
    #d("\n")

    d(f"------- SUMMARIES -----------")
    #summarise_applications(options[option], schools_data, applications)
    #summarise_placed(schools_data)

    chances = summarise_chances(option, pref_school, schools_data, fsm)

    percs = humanise_chances(pref_school, chances)

    disp = []
    for ind, school in enumerate(pref_school[:3]):
        disp.append(f"Your chance of placing at {school['school'].name} is {percs[ind]}%.")
    disp.append(f"Your chance of placing at no school and being assigned a fallback is {percs[3]}%.")

    d(disp)

    return disp

def summarise_chances(option, pref_school, schools_data, fsm=False):
    # chance of first school q0 = p(0)
    # chance of second school q1 = p(^0 & 1) = (1-p(0)) * p(1)
    # chance of third school q2 = p(^0 & ^1 & 2) = (1-p(0)) * (1-p(1)) * p(2)
    # chance of fall back q3 = 1-(q2 + q1 + q0)

    p = [0.0] * 3
    q = [0.0] * 4

    # calculate compound chance of NOT getting school and invert
    k_applicable = ['catchment_chance', 'outofcatchment_chance']
    if fsm:
        k_applicable.append('fsm_chance')
    for ind, school in enumerate(pref_school):
        no_chance = 1.0
        o(f"{school['school'].slug}")
        for k in k_applicable:
            chance = school[k]
            o(f"Chance for {k}: {chance}")
            no_chance *= (1 - chance)
        tot_chance = 1 - no_chance
        o(f"Overall no_chance = {no_chance} therefore chance is {tot_chance}")
        p[ind] = tot_chance

    q[0] = p[0]
    q[1] = (1-p[0]) * p[1]
    q[2] = (1-p[0]) * (1-p[1]) * p[2]
    q[3] = 1 - (q[2] + q[1] + q[0])

    return q

def nearest_perc_chunk(x, chunk=10):
    return int(chunk * round(x*100/chunk))

def humanise_chances(pref_school, chances):
    # clamp to nearest 10
    percs = [nearest_perc_chunk(c) for c in chances]
    if sum(percs) < 100:
        # optimism about last choice if total < 100 due to rounding
        percs[2] += 100 - sum(percs)

    return percs


@main.command()
@click.option('--option', required=True, type=click.Choice(['A', 'B', 'C', 'X'], case_sensitive=False))
@click.option('--year', required=True, type=click.Choice(['2024', '2026']))
@click.option('--geo_scaling', required=False, type=str, default=None)
@click.option('--popularity_scaling', required=False, type=str, default=None)
@click.option('--oversubscription_penalty', required=False, type=str, default=None)
@click.option('--damping', required=False, type=str, default=None)
@click.option('--schedule', required=False, type=str, default="linear")
@click.option('--tmax', required=False, type=float, default=500.0)
@click.option('--step_max', required=False, type=int, default=1000)
def anneal(option, year, geo_scaling, popularity_scaling, oversubscription_penalty, damping, schedule, tmax, step_max):
    # usage
    # anneal pref params to match actuals for 2024
    # use initial completely naive values
    # python catchsim.py anneal --option X --year 2024 --geo_scaling "1, 1, 1, 1, 1, 1, 1, 1, 1, 1" --popularity_scaling="1, 1, 1, 1, 1, 1, 1, 1, 1, 1" --oversubscription_penalty="1, 1, 1, 1, 1, 1, 1, 1, 1, 1" --tmax 500 --schedule quadratic --damping 0.05 --step_max 10000

    option = options[option]

    schools_data: dict = read_school_data('secondary_admissions_actuals_2425.csv', option, year, year)

    # vars to iterate
    # geo_scaling
    # popularity_scaling
    if geo_scaling:
        geo_scaling = [float(c.strip()) for c in geo_scaling.split(',')]
    else:
        geo_scaling = [school.geo_scaling for school_slug, school in schools_data.items()]
    if popularity_scaling:
        popularity_scaling = [float(c.strip()) for c in popularity_scaling.split(',')]
    else:
        popularity_scaling = [school.popularity_scaling for school_slug, school in schools_data.items()]
    if oversubscription_penalty:
        oversubscription_penalty = [float(c.strip()) for c in oversubscription_penalty.split(',')]
    else:
        oversubscription_penalty = [school.oversubscription_penalty for school_slug, school in schools_data.items()]

    x0 = geo_scaling + popularity_scaling + oversubscription_penalty

    if damping:
        damping = [float(c.strip()) for c in damping.split(',')]
    else:
        damping = [0.1] * 3
    # just 3 vals means use equal damping for all params
    if len(damping) == 1:
        damping = damping * len(x0)
    # just 3 vals means use equal damping for each param set
    if len(damping) == 3:
        damping = [damping[0]] * len(geo_scaling) \
                    + [damping[1]] * len(geo_scaling) \
                    + [damping[2]] * len(geo_scaling)
    print(damping)

    from anneal import minimize
    from functools import partial
    cost_func = partial(anneal_iterate, schools_data, option, year)
    # min max
    bounds = ((0.0001, 180.0),)*len(geo_scaling) \
                + ((0.0001, 30.0),)*len(geo_scaling) \
                + ((0.0001, 30.0),)*len(geo_scaling)
    mini = minimize(cost_func, x0, opt_mode='continuous', t_max=tmax, step_max=step_max,
                    bounds=bounds, cooling_schedule=schedule, damping=damping)
    mini.results()
    o("Final params")
    ns = len(schools_data)
    o("print geo:")
    o(mini.best_state[0:ns])
    o("print pop:")
    o(mini.best_state[ns:2*ns])
    o("print overpen:")
    o(mini.best_state[2*ns:])


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
        cost += abs(dist[school_slug][0] - school.first_prefs_received) / school.first_prefs_received
        cost += abs(dist[school_slug][1] - school.second_prefs_received) / school.second_prefs_received
        cost += abs(dist[school_slug][2] - school.third_prefs_received) / school.third_prefs_received
    return cost

def anneal_iterate(schools_data, option, year, x):
    # merge schools_data and scalings
    ns = len(schools_data)
    for c, (school_slug, school) in enumerate(schools_data.items()):
        school.geo_scaling = x[c]
        school.popularity_scaling = x[ns+c]
        school.oversubscription_penalty = x[2*ns+c]
    children = create_population(schools_data, option, year)
    cost = anneal_cost(schools_data, children)
    return cost


if __name__ == "__main__":
    main()

