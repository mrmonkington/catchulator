# CatchSIM

Work out a child's chances of being placed in a school according to Brighton and Hove
admissions process.

The stated process is:

  1. Looked after children
  2. Medical or other compelling reason
  3. Sibling link 
  4. FSM in catchment
  5. FSM not cactchment
  6. Catchment
  7. Other

Within each priority, a child receives a random number within the range -+1x10^9 which
is their "tie break" priority.

The places are then filled in 3 passes:

1st preference
2nd preference
3rd preference

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
