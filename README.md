# CppFlow

Cartesian path planning with IKFlow

Note: This project uses the `w,x,y,z` format for quaternions.

## Installation

python3.8 is required
```
poetry install
# note: you can do 'poetry install --without dev' to exclude some non essential functionality
```

## Getting started

Generate a plan for a single problem 
``` bash
# Problems:
#  - fetch_arm__circle
#  - fetch_arm__hello
#  - fetch_arm__rot_yz
#  - fetch_arm__s
#  - fetch_arm__square
#  - fetch__circle
#  - fetch__hello
#  - fetch__rot_yz
#  - fetch__s
#  - fetch__square
#  - panda__flappy_bird
#  - panda__2cubes
#  - panda__1cube

# you can replace 'fetch_arm__circle' with any of the problems above
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch_arm__circle --visualize
```

Recreate the results from the paper:
``` bash
python scripts/benchmark.py --planner_name=CppFlowPlanner
```