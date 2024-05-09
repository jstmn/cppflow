import numpy as np

PATH_TIME = 20

def update_path(filename):

    filepath = f"paths_torm/{filename}"
    with open(filepath, "r") as f:
        lines = [line.strip("\n") for line in f.readlines()]

    times = np.linspace(0, PATH_TIME, len(lines))

    with open(filename+".csv", "w", newline="\n") as f:
        f.write("time,x,y,z,qw,qx,qy,qz"+"\n")

        for idx, line in enumerate(lines):
            _, xyz, q = line.split(";")
            f.write(str(round(float(times[idx]), 8)) + "," + xyz+","+q+"\n")


for filename in ["circle", "hello", "rot_yz", "s", "square"]:
    update_path(filename)