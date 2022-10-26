from dataclasses import dataclass
from typing import List


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


@dataclass
class DetectorInference:
    yolo: List[DetectorObject]

    def __getitem__(self, item):
        return getattr(self, item, None)

    def __setitem__(self, item, value):
        setattr(self, item, value)
