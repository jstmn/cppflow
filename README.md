# CppFlow

Cartesian path planning with IKFlow. Open source implementation to the paper ["CppFlow: Generative Inverse Kinematics for Efficient and Robust Cartesian Path Planning"](https://arxiv.org/abs/2309.09102)

[![arxiv.org](https://img.shields.io/badge/cs.RO-%09arXiv%3A2309.09102-red)](https://arxiv.org/abs/2309.09102)


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
git checkout 2b6ad3097ad06af17e8d7eacdff78bbc98a1c3be
python scripts/benchmark.py --planner_name=CppFlowPlanner
```



## Citation

```
@INPROCEEDINGS{10611724,
    author={Morgan, Jeremy and Millard, David and Sukhatme, Gaurav S.},
    booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)}, 
    title={CppFlow: Generative Inverse Kinematics for Efficient and Robust Cartesian Path Planning}, 
    year={2024},
    volume={},
    number={},
    pages={12279-12785},
    keywords={Adaptation models;Generative AI;Graphics processing units;Kinematics;Programming;Trajectory;Planning},
    doi={10.1109/ICRA57147.2024.10611724}
}
```