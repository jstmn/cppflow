# header is not currently used.
std_msgs/Header header

# The name of the base link
string base_frame_id

# The name of the end effector link
string end_effector_frame_id

# Array of planning problems. CppFlow will solve all of them in parallel.
CppFlowProblem[] problems
