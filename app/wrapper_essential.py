from dataclasses import dataclass, fields
from typing import List
from pathlib import Path
import glob, re


@dataclass
class DetectorObject:
    """
    YOLO detector 객체로 사용하던 dict 를 감싼 오브젝트.
    보통의 dict 와 같이 obj["attr"] = value 처럼 사용할 수도 있다.
    """

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    confidence: float
    cls: int
    name: str

    depth: float

    def get_dict(self) -> dict:
        """객체의 dict 표현을 반환. obj["attr"] = value 와 같이 사용"""
        return self.__dict__

    def __getitem__(self, item):
        return getattr(self, item, None)

    def __setitem__(self, item, value):
        setattr(self, item, value)

    def bbox_coordinate_diagonal(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @staticmethod
    def from_dict(argDict: dict) -> "DetectorObject":
        fieldSet = {f.name for f in fields(DetectorObject) if f.init}
        filteredArgDict = {k: v for k, v in argDict.items() if k in fieldSet}
        return DetectorObject(**filteredArgDict)


@dataclass
class DetectorInference:
    yolo: List[DetectorObject]

    def __getitem__(self, item):
        return getattr(self, item, None)

    def __setitem__(self, item, value):
        setattr(self, item, value)


def increment_path(path, exist_ok=True, sep=""):
    # Increment path, i.e. runs/exp --> runs/exp{sep}0, runs/exp{sep}1 etc.
    path = Path(path)  # os-agnostic
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f"{path}{sep}{n}"  # update path
