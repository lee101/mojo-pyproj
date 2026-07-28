from enum import Enum


class TransformDirection(Enum):
    FORWARD = "FORWARD"
    INVERSE = "INVERSE"
    IDENT = "IDENT"


class ProjVersion(Enum):
    PROJ_4 = "PROJ_4"
    PROJ_5 = "PROJ_5"
