import re


def log_parser(filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()

    points = []
    for line in lines:
        finds = re.findall("Position\(x=(\d+\.\d+), y=(\d+\.\d+)", line)
        if finds:
            points.append(tuple(map(float, reversed(finds[0]))))

    return points
