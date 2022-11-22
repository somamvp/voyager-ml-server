"""app/runs/*.log 로그파일을 파싱해서 csv로 저장

Usage:
    python log_gps_parser.py --foldername app/runs
"""

import re
from collections import defaultdict
import pandas as pd

SESSION_DICT = defaultdict(lambda: [])

PARSED = []


def log_parser(filename: str):
    with open(filename, "r") as f:
        lines = f.readlines()

    session = None
    date_time = None
    seq_NO = None
    size = 10
    objects = None
    for line in lines:

        current_session = session_id(line)
        if current_session:
            session = current_session
            date_time = datetime(line)

        current_seq_NO = seq_no(line)
        if current_seq_NO:
            seq_NO = current_seq_NO

        point = gps(line)
        current_find_object = find_object(line)

        if current_find_object:
            objects = ",".join(map(str, current_find_object))

        if session and point and seq_NO and objects:
            SESSION_DICT[session].append(point)
            objects = ",".join(set(objects.split(",")))
            PARSED.append((date_time, session, seq_NO, size, *point, objects))

    df = pd.DataFrame(
        PARSED,
        columns=[
            "datetime",
            "session",
            "seq_NO",
            "size",
            "lat",
            "lon",
            "heading",
            "speed",
            "objects",
        ],
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def find_valid_first_match(
    pattern: str, line: str, prefix: str = "\| INFO     \|"
):
    finds = re.findall(f"{prefix}.*{pattern}", line)
    if finds:
        return finds[0]


def find_object(line: str):
    find = find_valid_first_match(
        "발견된 물체\(전체\):.(\[.*\])",
        line,
    )
    if find:
        return eval(find)


def datetime(line: str):
    return find_valid_first_match("(2022.+) \| INFO", line, prefix="")


def gps(line: str):
    find = find_valid_first_match(
        "Position\(x=(\d+\.\d+), y=(\d+\.\d+), heading=(\d+\.\d+), speed=(-?\d+(?:\.\d+)?)\)",
        line,
    )
    if find:
        position = [*map(float, find)]
        return (position[1], position[0], position[2], position[3])


def session_id(line: str):
    return find_valid_first_match("session_id (.+);", line)


def seq_no(line: str):
    return find_valid_first_match("SESSION: (.+) -", line)


def yolo_log(line: str):
    return find_valid_first_match(
        "(\d+)\s+([^\s]+)\s+(\[.*\])\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)",
        line,
        prefix="",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str)
    parser.add_argument("--foldername", type=str)
    args = parser.parse_args()

    if args.foldername:
        import os
        import glob

        filenames = glob.glob(os.path.join(args.foldername, "*.log"))
        for filename in filenames:
            print(filename)
            df = log_parser(filename)
    elif args.filename:
        df = log_parser(args.filename)

    print(f"total {len(SESSION_DICT)} session parsed")
    for key, points in SESSION_DICT.items():
        print(key, len(points))
    df.to_csv("parsed.csv", index=False)
