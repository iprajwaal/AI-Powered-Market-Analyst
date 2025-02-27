from enum import Enum, auto

class Name(Enum):
    GOOGLE_SEARCH = "google_search"
    INDUSTRY_REPORT = "industry_report"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    DATASET_SEARCH = "dataset_search"
    BRAINSTORM_USE_CASES = "brainstorm_use_cases"
    PRODUCT_SEARCH = "product_search"
    GOOGLE_TRENDS = "google_trends"
    NONE = "none"


    def __str__(self) -> str:
        return self.name.lower().replace("_", " ")
