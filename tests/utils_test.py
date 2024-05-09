import unittest

from cppflow.utils import calc_hash, TestSpecification


class UtilsTest(unittest.TestCase):
    def test_calc_hash(self):
        in_1 = {"verbosity": 0, "k": 25}
        in_2 = {"verbosity": 0, "k": 50}
        print(calc_hash(in_1), calc_hash(in_2))
        self.assertNotEqual(calc_hash(in_1), calc_hash(in_2))

        in_1 = {"verbosity": 0, "k": 35}
        in_2 = {"verbosity": 0, "k": 35}
        print(calc_hash(in_1), calc_hash(in_2))
        self.assertEqual(calc_hash(in_1), calc_hash(in_2))

        in_1 = {"verbosity": 0, "k": 45}
        in_2 = {"k": 45, "verbosity": 0}
        print(calc_hash(in_1), calc_hash(in_2))
        self.assertEqual(calc_hash(in_1), calc_hash(in_2))

    def test_TestSpecification(self):
        t1 = TestSpecification(planner="CppFlowPlanner", problem="panda__1cube", k=55, n_runs=3)
        t2 = TestSpecification(planner="CppFlowPlanner", problem="panda__1cube", k=55, n_runs=3)
        print(t1.get_hash(), t2.get_hash())
        self.assertEqual(t1.get_hash(), t2.get_hash())


if __name__ == "__main__":
    unittest.main()
