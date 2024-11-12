from typing import Dict, Type
import argparse
import sys
from datetime import datetime
import multiprocessing
import os
import socket
import psutil

import torch
import pandas as pd

from cppflow.planners import (
    PlannerSearcher,
    CppFlowPlanner,
    Planner,
    PlannerResult,
    TimingData,
    PlannerSettings
)
from cppflow.problem import problem_from_filename, get_all_problems, Problem
from cppflow.utils import set_seed
from cppflow.config import DEVICE
from cppflow.visualization import visualize_plan, plot_plan

torch.set_printoptions(linewidth=120)
# set_seed()

PLANNERS = {
    "CppFlowPlanner": CppFlowPlanner,
    "PlannerSearcher": PlannerSearcher,
}

BENCHMARKING_DIR = "/home/jstm/Projects/cppflowpaper2/benchmarking/cppflow"

PD_COLUMN_NAMES = [
    "Problem",
    "Robot",
    "Planner",
    "Valid plan",
    "time, total (s)",
    "time, ikflow (s)",
    "time, coll_checking (s)",
    "time, batch_opt (s)",
    "time, dp_search (s)",
    "time, optimizer (s)",
    "time per opt. step (s)",
    "Max positional error (mm)",
    "Max rotational error (deg)",
    "Mean positional error (mm)",
    "Mean rotational error (deg)",
    "Mjac - prismatic (cm)",
    "Mjac - revolute (deg)",
]


def _eval_planner_on_problem(planner: Type[Planner], problem: Problem, planner_settings: PlannerSettings):
    result = planner.generate_plan(problem, planner_settings)
    print()
    print(result.plan)
    print()
    print("timing:")
    print("  total:         ", result.timing.total)
    print("  ikflow:        ", result.timing.ikflow)
    print("  coll_checking: ", result.timing.coll_checking)
    print("  batch_opt:     ", result.timing.batch_opt)
    print("  dp_search:     ", result.timing.dp_search)
    print("  optimizer:     ", result.timing.optimizer)
    time_per_optimization_step = result.timing.optimizer / result.debug_info["n_optimization_steps"]
    round_amt = 5
    return [
        problem.fancy_name,
        problem.robot.name,
        planner.name,
        f"`{str(result.plan.is_valid).lower()}`",
        round(result.timing.total, 4),
        round(result.timing.ikflow, 4),
        round(result.timing.coll_checking, 4),
        round(result.timing.batch_opt, 4),
        round(result.timing.dp_search, 4),
        round(result.timing.optimizer, 4),
        round(time_per_optimization_step, 4),
        round(result.plan.max_positional_error_mm, round_amt),
        round(result.plan.mean_rotational_error_deg, round_amt),
        round(result.plan.mean_positional_error_mm, round_amt),
        round(result.plan.max_rotational_error_deg, round_amt),
        round(result.plan.mjac_cm, round_amt),
        round(result.plan.mjac_deg, round_amt),
    ], result.plan.is_valid


# TODO: update to only use LmFull. Or move this to a new script that's just for saving data to the paper repo
def eval_planners_on_problem(settings_dict: Dict, save_to_benchmarking: bool = True):
    """Run all planners on each problem"""
    # problems = get_all_problems()
    problems = [problem_from_filename("fetch_arm__square")]
    # problems = [problem_from_filename("panda__square"), problem_from_filename("fetch_arm__square")]
    planner_clcs = [CppFlowPlanner]

    markdown_filepath = f"scripts/problem_performance - {datetime.now().strftime('%m.%d-%H:%M')}.md"

    df_all = pd.DataFrame(columns=PD_COLUMN_NAMES)

    print("\n---------------------------------")

    for i, problem in enumerate(problems):
        print(f"\n\n{i} ======================\n")
        print(problem)

        df = pd.DataFrame(columns=PD_COLUMN_NAMES)

        for planner_clc in planner_clcs:
            planner = planner_clc(problem.robot)
            print("\n  ======\n")
            print(planner)
            new_row = _eval_planner_on_problem(planner, problem, settings_dict[planner.name])
            # df.loc[len(df)] = new_row
            # df_all.loc[len(df_all)] = new_row

        # print("\ndf:")
        # print(df)
        # df_succ = df[df["Valid plan"] != "`false`"].copy()
        # df_succ = df_succ.drop(["Valid plan", "Mean positional error", "Mean rotational error", "Problem"], axis=1)
        # df_fail = df[df["Valid plan"] == "`false`"].copy()
        # df_fail = df_fail.drop(["Valid plan", "Problem"], axis=1)

        # with open(markdown_filepath, "a") as f:
        #     if i == 0:
        #         cli_input = "python " + " ".join(sys.argv)
        #         dt = datetime.now().strftime("%m.%d-%H:%M:%S")
        #         f.write(f"# Planner results")
        #         f.write(f"\n\ndt: {dt} | cli_input: `{cli_input}`\n")
        #         f.write(f"\n\nparams:\n")
        #         for k, v in kwargs_dict.items():
        #             f.write(f"- {k}: `{v}`\n")
        #         f.write(f"\n\n")

        #     f.write(f"\n\n## Problem | **{problem.robot.name}, {problem.name}**")
        #     f.write(f"\n\nSucceeded:\n")
        #     f.write(df_succ.to_markdown())
        #     f.write(f"\n\nFailed:\n")
        #     f.write(df_fail.to_markdown())

    if save_to_benchmarking:
        assert psutil.cpu_count() == multiprocessing.cpu_count()
        now_str = datetime.now().strftime("%m.%d-%H:%M")
        df_all.to_csv(os.path.join(BENCHMARKING_DIR, f"results__{now_str}.csv"))
        with open(os.path.join(BENCHMARKING_DIR, f"results__{now_str}__params.md"), "a") as f:
            cli_input = "python " + " ".join(sys.argv)
            f.write(f"# Parameters")
            f.write(f"\n\ndt: {now_str} | cli_input: `{cli_input}`\n")
            f.write(f"\n\nparams:\n")
            for k, v in settings_dict.__dict__.items():
                if k[0] == "_":
                    continue
                f.write(f"- {k}: `{v}`\n")
            f.write(f"\n\n")
            f.write(f"\n\ncomputer:\n")
            f.write(f"- hostname: `{socket.gethostname()}`\n")
            f.write(f"- gpu: `{torch.cuda.get_device_name(device=DEVICE)}`\n")
            f.write(f"- #cpus: `{multiprocessing.cpu_count()}`\n")
            f.write(f"- ram size: `{psutil.virtual_memory().total / (1024*1024*1024)}` gb\n")
            f.write(f"\n\n")


def eval_planner_on_problems(planner_name: str, planner_settings: PlannerSettings):
    """Evaluate a planner on the given problems"""
    # problems = get_all_problems()
    problems = [
        problem_from_filename("panda__1cube"),
        problem_from_filename("fetch_arm__square"),
        problem_from_filename("fetch__square"),
    ]
    planners = [PLANNERS[planner_name](problem.robot) for problem in problems]

    df = pd.DataFrame(columns=PD_COLUMN_NAMES)
    succeeded = []
    failed = []

    for problem, planner in zip(problems, planners):
        print("\n---------------------------------")
        print(problem)
        new_row, is_valid = _eval_planner_on_problem(planner, problem, planner_settings)
        assert len(new_row) == len(PD_COLUMN_NAMES), (
            f"len(new_row)={len(new_row)} != len(PD_COLUMN_NAMES)={len(PD_COLUMN_NAMES)}\n. Column:"
            f" {PD_COLUMN_NAMES}\nrow: {new_row}"
        )
        df.loc[len(df)] = new_row

        # Update valid_dict
        if is_valid:
            succeeded.append(problem.full_name)
        else:
            failed.append(problem.full_name)

    df = df.sort_values(by=["Robot", "Problem"])
    df = df.drop(["Planner"], axis=1)

    dt = datetime.now().strftime("%d.%H:%M")
    with open(f"PLANNER_RESULTS - {planner_name} - {dt}.md", "w") as f:
        cli_input = "python " + " ".join(sys.argv)
        f.write(f"\n\n**{dt}** | Generated with `{cli_input}`")
        f.write(f"\n\nPlanner: **{planner.name}**")
        f.write(f"\n\nparams: `{planner_settings}`\n\n")
        f.write(df.to_markdown())

        valid_text = "\n```\n"
        valid_text += f"succeeded: {sorted(succeeded)}\n"
        valid_text += f"failed:    {sorted(failed)}\n"
        valid_text += "```"
        f.write(valid_text)

        # df_success
        f.write(f"\n\n**Successful plans**:\n\n")
        df_success = df[df["Valid plan"] != "`false`"].copy()
        df_success = df_success.drop(
            [
                "Valid plan",
                "Max positional error (mm)",
                "Max rotational error (deg)",
                "Mean positional error (mm)",
                "Mean rotational error (deg)",
            ],
            axis=1,
        )
        f.write(df_success.to_markdown())

        # df_failed
        f.write(f"\n\n**Failed plans**:\n\n")
        df_failed = df[df["Valid plan"] == "`false`"].copy()
        df_failed = df_failed.drop(["Valid plan"], axis=1)
        f.write(df_failed.to_markdown())

    print(df)


"""

Problems:
 - fetch_arm__circle
 - fetch_arm__hello
 - fetch_arm__rot_yz
 - fetch_arm__s
 - fetch_arm__square
 - fetch__circle
 - fetch__hello
 - fetch__rot_yz
 - fetch__s
 - fetch__square
 - panda__flappy_bird
 - panda__2cubes
 - panda__1cube

Example usage:

python scripts/evaluate.py --all_1 --planner CppFlowPlanner
python scripts/evaluate.py --all_2 --save_to_benchmarking

python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch_arm__circle --visualize
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch_arm__hello --visualize
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch_arm__rot_yz --visualize
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch_arm__rot_yz2 --visualize
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch_arm__s --visualize
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch_arm__square --visualize
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch__circle --visualize
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch__hello --visualize
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch__rot_yz --visualize
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch__rot_yz2 --visualize
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch__s --visualize
python scripts/evaluate.py --planner CppFlowPlanner --problem=fetch__square --visualize
"""



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner_name", type=str)
    parser.add_argument("--problem", type=str)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--plan_filepath", type=str)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--all_1", action="store_true")
    parser.add_argument("--all_2", action="store_true")
    parser.add_argument("--save_to_benchmarking", action="store_true")
    args = parser.parse_args()


    planner_settings_dict = {
        "CppFlowPlanner": PlannerSettings(
            k=175,
            do_rerun_if_large_dp_search_mjac=True,
            do_rerun_if_optimization_fails=True,
            do_return_search_path_mjac=True,
        ),
        "PlannerSearcher": PlannerSettings(
            k=175,
        ),
    }
    planner_settings = planner_settings_dict[args.planner_name]

    if args.problem is not None:
        problem = problem_from_filename(args.problem)
        print(problem)

        if args.plan_filepath is not None:
            plan = torch.load(args.plan_filepath)
            planner_result = PlannerResult(plan, TimingData(0, 0, 0, 0, 0, 0), [], [], {})
        else:
            planner: Planner = PLANNERS[args.planner_name](problem.robot)
            planner_result = planner.generate_plan(problem, planner_settings)

            # save results to disk
            # torch.save(planner_result.plan, f"pt_tensors/plan__{problem.full_name}__{planner.name}.pt")
            # df = pd.DataFrame(planner_result.plan.q_path.cpu().numpy())
            # df.to_csv(f"pt_tensors/plan__{problem.full_name}__{planner.name}.csv", index=False)

            print()
            print(planner_result.plan)
            print()
            print("timing:")
            print("  total:         ", planner_result.timing.total)
            print("  ikflow:        ", planner_result.timing.ikflow)
            print("  coll_checking: ", planner_result.timing.coll_checking)
            print("  batch_opt:     ", planner_result.timing.batch_opt)
            print("  dp_search:     ", planner_result.timing.dp_search)
            print("  optimizer:     ", planner_result.timing.optimizer)
            print()
            print("debug_info:")
            for k, v in planner_result.debug_info.items():
                print(f"  {k}: {v}")

        if args.plot:
            plot_plan(planner_result.plan, problem, planner_result.other_plans, planner_result.other_plans_names)
        if args.visualize:
            visualize_plan(planner_result.plan, problem, start_delay=3)

    if args.all_1:
        eval_planner_on_problems(args.planner_name, planner_settings)

    elif args.all_2:
        eval_planners_on_problem(planner_settings_dict, args.save_to_benchmarking)
