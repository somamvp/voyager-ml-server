YOLOV5_PT_FILE = "basicv5.pt"
YOLOV7_BASIC_PT_FILE = "wesee7_6.pt"
YOLOV7_DESC_PT_FILE = "extended_4.pt"
YOLO_IDX_TO_NAME = [
    "Zebra_Cross",
    "R_Signal",
    "G_Signal",
    "Braille_Block",
    "person",
    "dog",
    "tree",
    "car",
    "bus",
    "truck",
    "motorcycle",
    "bicycle",
    "none",
    "wheelchair",
    "stroller",
    "kickboard",
    "bollard",
    "manhole",
    "labacon",
    "bench",
    "barricade",
    "pot",
    "table",
    "chair",
    "fire_hydrant",
    "movable_signage",
    "bus_stop",
]
YOLO_NAME_TO_KOREAN = {
    "Zebra_Cross": "횡단보도",
    "R_Signal": "빨간신호",
    "G_Signal": "초록신호",
    "Braille_Block": "점자블록",
    "person": "사람",
    "dog": "강아지",
    "tree": "가로수",
    "car": "차",
    "bus": "버스",
    "truck": "트럭",
    "motorcycle": "오토바이",
    "bicycle": "자전거",
    "none": "",
    "wheelchair": "휠체어",
    "stroller": "유모차",
    "kickboard": "킥보드",
    "bollard": "볼라드",
    "manhole": "맨홀",
    "labacon": "라바콘",
    "bench": "벤치",
    "barricade": "바리케이드",
    "pot": "화단",
    "table": "탁자",
    "chair": "의자",
    "fire_hydrant": "소화전",
    "movable_signage": "입간판",
    "bus_stop": "버스정류장",
}
YOLO_NAME_TO_IDX = {name: idx for idx, name in enumerate(YOLO_IDX_TO_NAME)}
YOLO_IDX_TO_KOREAN = {
    idx: korean for idx, korean in enumerate(list(YOLO_NAME_TO_KOREAN.values()))
}
YOLO_THRES = {
    "Zebra_Cross": 0.6,
    "R_Signal": 0.4,
    "G_Signal": 0.4,
    "Braille_Block": 0.4,
    "person": 0.5,
    "dog": 0.4,
    "tree": 0.35,
    "car": 0.5,
    "bus": 0.4,
    "truck": 0.4,
    "motorcycle": 0.35,
    "bicycle": 0.35,
    "none": 1,
    "wheelchair": 0.3,
    "stroller": 0.3,
    "kickboard": 0.3,
    "bollard": 0.4,
    "manhole": 0.5,
    "labacon": 0.45,
    "bench": 0.4,
    "barricade": 0.4,
    "pot": 0.35,
    "table": 0.35,
    "chair": 0.35,
    "fire_hydrant": 0.4,
    "movable_signage": 0.4,
    "bus_stop": 0.4,
}
YOLO_OBS_TYPE = {
    "MOVING": [
        "person",
        "dog",
        "car",
        "bus",
        "truck",
        "motorcycle",
        "bicycle",
        "wheelchair",
        "stroller",
        "kickboard",
    ],
    "STATIC": [
        "tree",
        "bollard",
        "labacon",
        "bench",
        "barricade",
        "pot",
        "table",
        "chair",
        "fire_hydrant",
        "movable_signage",
    ],
    "FLOOR": ["Zebra_Cross", "Braille_Block", "manhole"],
    "BUS_STOP": ["bus_stop"],
    "LIGHTS": ["R_Signal", "G_Signal"],
}
