from dataclasses import dataclass
import csv

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

#@dataclass
#class Child
#    pref_1: School
#    pref_2: School
#    pref_3: School

@dataclass
class Catchment:
    name: str
    # total of all schools in catchment
    pan: int
    # how many children live in catchment
    apps_actual: int
    # how many children actually qualify for FSM in this area 
    fsm_actual: int
    # geometry
    # geo: 

@dataclass
class LA:
    name: str
    pan: int # total of all schools in LA
    apps_expected: int
    fsm_expected: int


def read_school_data(filename):
    schools = []
    with open(filename, newline='') as file_obj:
        reader_object = csv.DictReader(file_obj)
        for row in reader_object:
            school = School(
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
            schools.append(school)
    return schools

schools_data = read_school_data('secondary_admissions_actuals_2425.csv')

for school in schools_data:
    print(school.name, school.pan, school.total_places, school.offers_made, school.offers_accepted)

def 
