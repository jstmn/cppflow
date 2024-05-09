import unittest

import torch

from cppflow.utils import set_seed, to_torch
from cppflow.problem import problem_from_filename, Problem
from cppflow.collision_detection import (
    get_only_non_colliding_qpaths,
    qpaths_batched_env_collisions,
    qpaths_batched_self_collisions,
    self_colliding_configs_capsule,
    env_colliding_configs_capsule,
)


set_seed()


def _only_non_colliding_gt(problem: Problem, qpaths: torch.Tensor):
    non_self_colliding = [qp for qp in qpaths if not self_colliding_configs_capsule(problem, qp).any()]
    print("# non-self_colliding: ", len(non_self_colliding), "/", len(qpaths))
    safe = [qp for qp in non_self_colliding if not env_colliding_configs_capsule(problem, qp).any()]
    print("# safe:               ", len(safe), "/", len(qpaths))
    return safe


class CollisionCheckingTest(unittest.TestCase):
    def test_qpaths_filtered_the_same(self):
        n = 5
        k = 50
        p_old = problem_from_filename("panda__1cube")
        problem = Problem(
            target_path=p_old.target_path[0:n],
            robot=p_old.robot,
            name=p_old.name,
            full_name=p_old.full_name,
            obstacles=p_old.obstacles,
            obstacles_Tcuboids=p_old.obstacles_Tcuboids,
            obstacles_cuboids=p_old.obstacles_cuboids,
            obstacles_klampt=p_old.obstacles_klampt,
        )
        ikflow_qpaths = [to_torch(problem.robot.sample_joint_angles(n)) for _ in range(k)]

        # Ground truth
        filtered_gt = _only_non_colliding_gt(problem, ikflow_qpaths)
        assert len(filtered_gt) > 0

        # Code to test
        q = torch.stack(ikflow_qpaths)
        self_collision_violations = qpaths_batched_self_collisions(problem, q)
        env_collision_violations = qpaths_batched_env_collisions(problem, q)
        returned = get_only_non_colliding_qpaths(ikflow_qpaths, self_collision_violations, env_collision_violations)

        # Check
        self.assertEqual(len(filtered_gt), len(returned))
        for qp1, qp2 in zip(filtered_gt, returned):
            self.assertTrue(torch.allclose(qp1, qp2))


if __name__ == "__main__":
    unittest.main()
