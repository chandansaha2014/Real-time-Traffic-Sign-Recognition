import enum


class SuperclassType(enum.Enum):
    _00_All = 0
    _01_Prohibitory = 1
    _02_Warning = 2
    _03_Mandatory = 3
    _04_Other = 4


class ClassifierType(enum.Enum):
    logit = 1
    svm = 2


class ModelType(enum.Enum):
    _00_undefined = 0
    _01_conv2_mlp2 = 1
    _02_conv3_mlp2 = 2
    _03_conv3_mlp3 = 3
