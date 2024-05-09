
# Problems 
Problems are directly copied from 'TORM's github repo (https://github.com/cheulkang/TORM/tree/main/src/data/problem) so that CppFlow will have a fair comparison against the other authors work.

problem name | includes obstacles?
-------------|--------------------
fetch_circle | yes
fetch_hello | no
fetch_rotation | no
fetch_s_two | yes
fetch_s | yes
fetch_square | yes



## Notes
- The difference between the 's' and 's_two' paths is unclear. Also unclear is which one was used in the paper. 

## Path offsets
For all problems with the fetch robot, the `path_<xyz/R>_offset` value is copied from the TORM repository. Note that the offset is named 'start_pose' there (see https://github.com/cheulkang/TORM/blob/main/src/data/problem/fetch_square.yaml#L8).
 
