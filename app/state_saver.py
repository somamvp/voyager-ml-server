from dataclasses import dataclass, fields
import pickle, json

from typing import Any, Optional

from app.state_machine import StateMachine
from app.tracking import TrackerWrapper
from app.description import ClockCycleStateActivator


def pickle_string_2_obj(string: str):
    return pickle.loads(string.encode("latin-1"))


def obj_2_pickle_string(obj: Any):
    return pickle.dumps(obj).decode("latin-1")


@dataclass
class StateSaver:
    state_machine: Optional[StateMachine] = None
    tracker: Optional[TrackerWrapper] = None
    clock_activator: Optional[ClockCycleStateActivator] = None

    @staticmethod
    def from_dict(argDict: dict) -> "StateSaver":
        fieldSet = {f.name for f in fields(StateSaver) if f.init}
        filteredArgDict = {k: v for k, v in argDict.items() if k in fieldSet}
        return StateSaver(**filteredArgDict)

    def stringify(self) -> str:
        field_set = {f.name for f in fields(StateSaver) if f.init}
        self_dict = {field: getattr(self, field) for field in field_set}
        string_dict = {
            field: obj_2_pickle_string(obj) for field, obj in self_dict.items()
        }
        return json.dumps(string_dict)

    @staticmethod
    def unstringify(string: str) -> "StateSaver":
        string_dict = json.loads(string, strict=False)
        self_dict = {
            field: pickle_string_2_obj(s) for field, s in string_dict.items()
        }
        return StateSaver.from_dict(self_dict)

    def get_dict(self) -> dict:
        """객체의 dict 표현을 반환. obj["attr"] = value 와 같이 사용"""
        return self.__dict__

    def __getitem__(self, item):
        return getattr(self, item, None)

    def __setitem__(self, item, value):
        setattr(self, item, value)
