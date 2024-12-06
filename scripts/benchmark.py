from time import time
import sys
from datetime import datetime
from io import StringIO
from warnings import warn
import argparse
import os

import torch
import pandas as pd

from cppflow.planners import PlannerSearcher, CppFlowPlanner
from cppflow.problem import ALL_PROBLEM_FILENAMES, get_problem_dict
from cppflow.utils import set_seed
from cppflow.config import SELF_COLLISIONS_IGNORED, ENV_COLLISIONS_IGNORED, DEBUG_MODE_ENABLED

RESULTS_CSV_COLS = (
    "Time Elapsed (s)",
    "is valid",
    "Mean Pos Error (mm)",
    "Max Pos Error (mm)",
    "Mean Rot Error (deg)",
    "Max Rot Error (deg)",
    "Mjac (deg)",
    "Mjac (cm)",
    "pct_self-colliding",
    "pct_env-colliding",
    "path_length_rad",
    "path_length_m",
)


torch.set_printoptions(linewidth=120)
set_seed()

PLANNERS = {
    "CppFlowPlanner": CppFlowPlanner,
    "PlannerSearcher": PlannerSearcher,
}


def main():
    n_files = len([
        item
        for item in os.listdir("scripts/benchmarking_output")
        if os.path.isfile(os.path.join("scripts/benchmarking_output", item))
    ])
    assert n_files == 1, f"Expected 1 file in 'scripts/benchmarking_output/' but found {n_files}."
    assert SELF_COLLISIONS_IGNORED == ENV_COLLISIONS_IGNORED == DEBUG_MODE_ENABLED == False

    parser = argparse.ArgumentParser()
    parser.add_argument("--planner_name", type=str, required=True)
    args = parser.parse_args()
    planner_name = args.planner_name

    # 0.034671815s/step, 0.07943 s to log. 60s -> 1813 steps -> 1813*0.07943=2.4min for logging
    # --> 44min / rerun

    n_reruns = 10
    # n_reruns = 3
    # n_reruns = 1

    kwargs_dict = {
        "CppFlowPlanner": {
            "k": 175,
            "verbosity": 0,
            "run_batch_opt": False,
            "do_rerun_if_large_dp_search_mjac": True,
            "do_rerun_if_optimization_fails": True,
            # "only_1st": True,
            "only_1st": False,
        },
    }
    assert not kwargs_dict[planner_name][
        "only_1st"
    ], "'only_1st' returns after the first trajectory is found. you probably don't want to enable this"

    problems = ALL_PROBLEM_FILENAMES
    # problems = ["fetch_arm__s", "fetch_arm__circle"]
    # problems = ["fetch_arm__s"]
    # problems = ["fetch__s"]
    problem_dict = get_problem_dict(problems)

    for _, problem in problem_dict.items():
        robot_name = problem.robot.name
        problem_name = problem.name
        for i in range(n_reruns):
            print("\n===========================================================================================")
            print(f"  ===           {planner_name}\t| {robot_name} - {problem_name}\t{i+1}/{n_reruns}           ===\n")
            problem = problem_dict[f"{robot_name}__{problem_name}"]
            planner = PLANNERS[planner_name](problem.robot)
            print(problem)
            results_df = {"df": pd.DataFrame(columns=RESULTS_CSV_COLS), "t0": time()}
            planner_result = planner.generate_plan(problem, **kwargs_dict[planner_name], results_df=results_df)
            print(planner_result.plan)

            # Save result to benchmarking_output/
            results_filepath = (
                f"scripts/benchmarking_output/cppflow__{planner_name}__{robot_name}__{problem_name}__results.csv"
            )
            s = StringIO()
            results_df["df"].to_csv(s)
            with open(results_filepath, "a") as f:
                f.write("\n\n==========\n")
                f.write(s.getvalue())
                f.write("==========\n")
            print(results_df["df"])

    # Write kwargs as well for record keeping
    markdown_filepath = (
        f"scripts/benchmarking_output/cppflow__{planner_name}__{datetime.now().strftime('%m.%d-%H:%M')}.md"
    )
    with open(markdown_filepath, "w") as f:
        dt_str = datetime.now().strftime("%m.%d-%H:%M:%S")
        cli_input = "python " + " ".join(sys.argv)
        f.write(f"# Planner results")
        f.write(f"\n\ndt: {dt_str} | cli_input: `{cli_input}`\n")
        f.write(f"\n\nparams:\n")
        for k, v in kwargs_dict.items():
            f.write(f"- {k}: `{v}`\n")
        f.write(f"\n\n")


"""
python scripts/benchmark.py --planner_name=CppFlowPlanner
"""

if __name__ == "__main__":
    warn("Ensure no other compute processes are running on the machine - this includes jupyter notebooks.")
    main()
