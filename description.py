# Input ex) [{name:'R_Signal', xmin, xmax, ymin, ymax, conf:0.8, depth:200},{...}] 
from enum import Enum
from dataclasses import dataclass
from sre_constants import RANGE

RANGE_TO_ALPHA = 255/20

class ReadMode(Enum):
    LEFT_TO_RIGHT = 1
    RIGHT_TO_LEFT = 2
    NEAR_TO_FAR = 3

# 예상되는 문제1 너무 많은 객체 읽게 되는 경우
# 예상되는 문제2 적절한 depth range값을 찾기 어려움


# Depth  픽셀값0~255 : 거리0~20 : 실제사용0~7
def inform(data, mode:int = ReadMode.LEFT_TO_RIGHT, range:float = 5.0):
    range_in_alpha = range * RANGE_TO_ALPHA
    guide_list = []

    for el in data:
        if el['depth']<range_in_alpha:
            el['xc'] = (el['xmin']+el['xmax'])/2
            el['yc'] = (el['ymin']+el['ymax'])/2
            guide_list.append(el)

    if mode==1:
        return sorted(guide_list, key=lambda x: x['xc'], reverse=False)
    elif mode==2:
        return sorted(guide_list, key=lambda x: x['xc'], reverse=True)
    elif mode==3:
        return sorted(guide_list, key=lambda x: x['depth'], reverse=False)
    else:
        print("Description mode selected incorrectly")

    return guide_list