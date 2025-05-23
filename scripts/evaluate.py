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

from cppflow.planners import PlannerSearcher, CppFlowPlanner, Planner
from cppflow.data_types import PlannerResult, TimingData, PlannerSettings, Constraints, Problem
from cppflow.data_type_utils import problem_from_filename
from cppflow.utils import set_seed, to_torch
from cppflow.config import DEVICE, SELF_COLLISIONS_IGNORED, ENV_COLLISIONS_IGNORED, DEBUG_MODE_ENABLED
from cppflow.visualization import visualize_plan, plot_plan
from cppflow.collision_detection import qpaths_batched_self_collisions, qpaths_batched_env_collisions

torch.set_printoptions(linewidth=120)
set_seed()

PLANNERS = {
    "CppFlow": CppFlowPlanner,
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

CONSTRAINTS = Constraints(
    max_allowed_position_error_cm=0.01,  # 0.1mm
    max_allowed_rotation_error_deg=0.1,
    max_allowed_mjac_deg=7.0,  # from the paper
    max_allowed_mjac_cm=2.0,  # from the paper
)


def _eval_planner_on_problem(planner: Type[Planner], problem: Problem, planner_settings: PlannerSettings):
    result = planner.generate_plan(problem, planner_settings)
    print()
    print(result.plan)
    print()
    print(result.timing)
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
    planner_clcs = [CppFlowPlanner]
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
            df_all.loc[len(df_all)] = new_row

    if save_to_benchmarking:
        assert psutil.cpu_count() == multiprocessing.cpu_count()
        now_str = datetime.now().strftime("%m.%d-%H:%M")
        df_all.to_csv(os.path.join(BENCHMARKING_DIR, f"results__{now_str}.csv"))
        with open(os.path.join(BENCHMARKING_DIR, f"results__{now_str}__params.md"), "a") as f:
            cli_input = "uv run python " + " ".join(sys.argv)
            f.write("# Parameters")
            f.write(f"\n\ndt: {now_str} | cli_input: `{cli_input}`\n")
            f.write("\n\nparams:\n")
            for k, v in settings_dict.__dict__.items():
                if k[0] == "_":
                    continue
                f.write(f"- {k}: `{v}`\n")
            f.write("\n\n")
            f.write("\n\ncomputer:\n")
            f.write(f"- hostname: `{socket.gethostname()}`\n")
            f.write(f"- gpu: `{torch.cuda.get_device_name(device=DEVICE)}`\n")
            f.write(f"- #cpus: `{multiprocessing.cpu_count()}`\n")
            f.write(f"- ram size: `{psutil.virtual_memory().total / (1024*1024*1024)}` gb\n")
            f.write("\n\n")


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
        cli_input = "uv run python " + " ".join(sys.argv)
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
        f.write("\n\n**Successful plans**:\n\n")
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
        f.write("\n\n**Failed plans**:\n\n")
        df_failed = df[df["Valid plan"] == "`false`"].copy()
        df_failed = df_failed.drop(["Valid plan"], axis=1)
        f.write(df_failed.to_markdown())

    print(df)


def get_initial_configuration(problem: Problem):
    for _ in range(25):
        initial_configuration = to_torch(
            problem.robot.inverse_kinematics_klampt(problem.target_path[0].cpu().numpy())
        ).view(1, 1, problem.robot.ndof)
        if not (
            qpaths_batched_env_collisions(problem, initial_configuration)
            or qpaths_batched_self_collisions(problem, initial_configuration)
        ):
            print(f"Initial configuration {initial_configuration} is collision free")
            return initial_configuration.view(1, problem.robot.ndof)
    raise RuntimeError("Could not find collision free initial configuration")


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

uv run python scripts/evaluate.py --all_1 --planner CppFlow
uv run python scripts/evaluate.py --all_2 --save_to_benchmarking

uv run python scripts/evaluate.py --planner CppFlow --problem=fetch_arm__circle --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=fetch_arm__hello --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=fetch_arm__rot_yz --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=fetch_arm__rot_yz2 --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=fetch_arm__s --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=fetch_arm__square --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=fetch__circle --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=fetch__hello --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=fetch__rot_yz --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=fetch__rot_yz2 --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=fetch__s --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=fetch__square --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=panda__1cube --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=panda__2cubes --visualize
uv run python scripts/evaluate.py --planner CppFlow --problem=panda__flappy_bird --visualize

uv run python scripts/evaluate.py --planner CppFlow --problem=fetch_arm__hello_mini --visualize --use_fixed_initial_configuration

uv run python scripts/evaluate.py --planner CppFlow --problem=panda__1cube_mini --plan_filepath=many_env_collisions[0].pt
uv run python scripts/evaluate.py --planner CppFlow --problem=panda__1cube_mini --plot --use_fixed_initial_configuration
uv run python scripts/evaluate.py --planner CppFlow --problem=panda__1cube_mini
"""


def main(args):
    planner_settings_dict = {
        "CppFlow": PlannerSettings(
            verbosity=2,
            k=175,
            tmax_sec=5.0,
            anytime_mode_enabled=False,
            do_rerun_if_large_dp_search_mjac=True,
            do_rerun_if_optimization_fails=False,
            do_return_search_path_mjac=False,
        ),
        "CppFlow_fixed_q0": PlannerSettings(
            verbosity=2,
            k=175,
            tmax_sec=3.0,
            anytime_mode_enabled=False,
            latent_vector_scale=0.5,
            do_rerun_if_large_dp_search_mjac=False,
            do_rerun_if_optimization_fails=False,
            do_return_search_path_mjac=False,
        ),
        "PlannerSearcher": PlannerSettings(
            k=175,
            tmax_sec=5.0,
            anytime_mode_enabled=False,
        ),
    }
    planner_settings = (
        planner_settings_dict[args.planner_name]
        if not args.use_fixed_initial_configuration
        else planner_settings_dict[args.planner_name + "_fixed_q0"]
    )

    if args.problem is not None:
        problem = problem_from_filename(CONSTRAINTS, args.problem)
        print(problem)
        print(problem.constraints)

        if args.use_fixed_initial_configuration:
            problem.initial_configuration = get_initial_configuration()

        if args.plan_filepath is not None:
            plan = torch.load(args.plan_filepath)
            planner_result = PlannerResult(plan, TimingData(0, 0, 0, 0, 0, 0), [], [], {})
        else:
            planner: Planner = PLANNERS[args.planner_name](planner_settings, problem.robot)
            planner_result = planner.generate_plan(problem)

            # save results to disk
            # torch.save(planner_result.plan, f"pt_tensors/plan__{problem.full_name}__{planner.name}.pt")
            # df = pd.DataFrame(planner_result.plan.q_path.cpu().numpy())
            # df.to_csv(f"pt_tensors/plan__{problem.full_name}__{planner.name}.csv", index=False)

            print()
            print("   ======   Planner result   ======")
            print()
            print(planner_result.plan)
            print()
            print(planner_result.timing)
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


if __name__ == "__main__":
    assert SELF_COLLISIONS_IGNORED == ENV_COLLISIONS_IGNORED == DEBUG_MODE_ENABLED == False

    parser = argparse.ArgumentParser()
    parser.add_argument("--planner_name", type=str)
    parser.add_argument("--problem", type=str)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--plan_filepath", type=str)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--all_1", action="store_true")
    parser.add_argument("--all_2", action="store_true")
    parser.add_argument("--save_to_benchmarking", action="store_true")
    parser.add_argument("--use_fixed_initial_configuration", action="store_true")
    args = parser.parse_args()

    main(args)
