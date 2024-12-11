from enum import Enum, auto


class Name(Enum):
    GOOGLE_SEARCH = auto()
    INDUSTRY_REPORT = auto()
    COMPETITOR_ANALYSIS = auto()
    DATASET_SEARCH = auto()
    BRAINSTORM_USE_CASES = auto()
    PRODUCT_SEARCH = auto()
    GOOGLE_TRENDS = auto()
    NONE = auto()


    def __str__(self) -> str:
        return self.name.lower().replace("_", " ")
