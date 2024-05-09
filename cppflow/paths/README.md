

Paths are from the 'TORM' github repo: https://github.com/cheulkang/TORM/tree/main/src/data/scene

These paths (I believe) use a format `1.0/0.0;x,y,z;qw,qx,qy,qz`, where the `1.0/0.0` indicates that the first value is either a 1 or a 0. I don't know what this value represents.

> Note: I'm pretty sure the quaternions in the torm paths are in `w,x,y,z` format. The quaternions are created like `tf::Quaternion q(input_rots[i][1], input_rots[i][2], input_rots[i][3], input_rots[i][0])` (see https://github.com/cheulkang/TORM/blob/main/src/torm_problem.cpp#L122). The definition for `tf::Quaternion` is `Quaternion(const tfScalar& x, const tfScalar& y, const tfScalar& z, const tfScalar& w)` (defined here https://github.com/ros/geometry/blob/noetic-devel/tf/include/tf/LinearMath/Quaternion.h#L39). This indicates the quaternions in paths_torm/ are saved in `w,x,y,z` format.  

I have updated these path files to .csv files, with the format `x,y,z,qw,qx,wy,wz`