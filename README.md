# CatchSIM

Work out a child's chances of being placed in a school according to Brighton and Hove
admissions process, according to one of the following models:

   - X. The current arrangements
   - A. Option A from the pre-engagement proposals
     - Catchment tweaks + PAN reduction set P
   - B. Option B from pre-eng
     - Large catchments separating DS and V + PAN set P
   - C. Option C from pre-eng
     - Single school catchments + PAN set P
   - Z. Option Z the proposal possibly being taken to consultation
     - Minor catchment tweak, new 20% out of catchment priority, PAN set Q

The stated process for community schools and other schools for which the LA is the admission authority:

  1. Looked after children
  2. Medical or other compelling reason
  3. Sibling link 
  4. FSM in catchment
  5. FSM not catchment
  6. (only in Z) 20% reservation for out of catchment applications, from children in catchments with a single school
  6. Other in-catchment children
  7. All other applicants

Within each priority, a child receives a random number within the range -+1x10^9 which
is their "tie break" priority.

The process for non-LA admissions schools, currently just the two faith schools (CN and Kings) is more complex, but we do know tthat they fill almost entirely with 1st prefs and from quite widely in the LA.

CN:

  - Looked after catholics
  - Catholics (mostly from feeder schools)
  - Other Christians (mostly from feeder schools)
  - Other faiths
  - Other children

The tie break for both faith schools is distance, but there is no limit for each priority so in effect it doesn't matter.

Kings:

  - Looked after christians
  - Christians from feeder schools
  - Other Christians @ 50% of remaining places
  - Other children @ 50% of remaining places

The tie-break is distance, and importantly from two nodal points. This model currently just considers the first nodal point, the school's location.

The places are then filled in 3 or 4 passes (depending on the chosen model):

 - 1st preference
 - 2nd preference
 - 3rd preference
 - (only in Z) 4th preference

Between each pass, children are removed from the pool and the priorities re-run.

## Running

Start nominatin server with B&H OSM data using eithe docker or podman:

```
# persist nominatim data
mkdir nominatim-data
[docker|podman] run -it \
  -e PBF_URL=https://download.geofabrik.de/europe/united-kingdom/england/west-sussex-latest.osm.pbf \
  -p 8080:8080 \
  --name nominatim \
  -e IMPORT_GB_POSTCODES=true \
  -v `pwd`/nominatim-data:/var/lib/postgresql/14/main:z
  mediagis/nominatim:4.4
```

Note:
 - Need west sussex data for nominatim because for some reason Brighton is in that!

Activate a python3 env (e.g. using `python -mvenv .env && . .env/bin/activate`) and install deps

```
pip install -r requirements.txt
```

WIP but something like this for prototype
```
python catchsim.py sim --postcode "BN1 1AA" --option [A,B,C,X,Z] --prefs varndean stringer patcham [optional 4th pref for Z] --panyear 2024 --popyear 2024
```

To run the simulated annealing in order to tune preference params, use
```
python catchsim.py anneal --option [A,B,C] --year 2024
```

It will try to find params that generate a similar distribution of preferences to the 2024 actuals. Plug the geo and pop numbers into `secondary_admissions_actuals_2425.csv`.
