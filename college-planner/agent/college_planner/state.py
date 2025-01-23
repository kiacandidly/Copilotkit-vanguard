from typing import Literal, TypedDict, List, Optional, Dict
from langgraph.graph import MessagesState
from pydantic import BaseModel
class Place(BaseModel):
    """A place."""
    id: str
    name: str
    address: str
    latitude: float
    longitude: float
    rating: float
    description: Optional[str]

class Trip(BaseModel):
    """A trip."""
    id: str
    name: str
    center_latitude: float
    center_longitude: float
    zoom: int # 13 for city, 15 for airport
    places: List[Place]

class SearchProgress(BaseModel):
    """The progress of a search."""
    query: str
    results: list[str]
    done: bool

class PlanningProgress(BaseModel):
    """The progress of a planning."""
    trip: Trip
    done: bool


class College(TypedDict):
    """A college."""
    school_id: int
    school_name: str
    school_city: str
    school_state: str
    school_zip: str
    school_url: str
    latitude: float
    longitude: float
    is_public: bool
    hbcu: bool
    women_only: bool
    religious_affiliation: str
    enrollment_size: int
    admission_rate_pct: float
    in_state_tuition_and_fees: int
    out_of_state_tuition_and_fees: int
    net_price_0_30k: int
    net_price_30k_48k: int
    net_price_48k_75k: int
    net_price_75k_110k: int
    net_price_110k_and_up: int
    mean_earnings_6_yrs_after_entry: int


# class College(BaseModel):
#     """A College."""
#     school_id: int
#     latitude: float
#     longitude: float
#     school_city: str
#     school_zip: str
#     school_name: str
#     school_state: str
#     is_public: bool
#     hbcu: bool
#     women_only: bool
#     religious_affiliation: str
#     enrollment_size: int
#     admission_rate_pct: float
#     academics: str
#     athletics: str
#     professors: str
#     dorms: str
#     campus_food: str
#     student_life: str
#     safety: str
#     mean_earnings_6_yrs_after_entry: int
#     housing_cost: int
#     average_cost_of_attendance: int
#     in_state_tuition_and_fees: int
#     out_of_state_tuition_and_fees: int
#     net_price_0_30k: int
#     net_price_30k_48k: int
#     net_price_48k_75k: int
#     net_price_75k_110k: int
#     net_price_110k_and_up: int
#     average_net_price: int
#     average_net_aid: int
#     school_url: str

class CollegeProgress(BaseModel):
    """The progress of a college search."""
    query: str
    # results: list[College]
    done: bool

class AgentState(MessagesState):
    """The state of the agent."""
    college_progress: List[CollegeProgress]
    selected_trip_id: List[str]
    trips: List[Trip]
    search_progress: List[SearchProgress]
    planning_progress: List[PlanningProgress]


class Major(BaseModel):
    """A major."""
    school_id: int
    major: str
    major_category: str
    class_size: int
    median_earnings_4_yrs: int
    school_name: str

class Scholarship(BaseModel):
    """A scholarship."""
    id: int
    award_amount: float
    scholarship_name: str
    description: str
    application_link: str
    application_open_date: str
    application_close_date: str
    merit_based: bool
    need_based: bool
    essay_required: bool
    eligible_majors: List[str]
    eligible_academic_years: List[str]
    eligible_gpa: float
    eligible_states: List[str]
    eligible_regions: str
    eligibility_other: List[str]


class Portfolio(BaseModel):
    """A portfolio."""
    id: int
    type: Literal["age_based", "static", "individual", "other"]
    age_based_year: int
    expense_ratio: int
    equity_allocation: float
    yr_1_performance: float | None
    yr_3_performance: float | None
    yr_5_performance: float | None
    yr_10_performance: float | None
    
class InvestmentAccounts(BaseModel):
    """College Investment account."""
    id: int
    name: str
    state: str
    tax_deduction: int 
    type: Literal["direct", "advisor"]
    max_contribution: int
    min_contribution: int
    enrollment_fee: int
    expense_ratio: int
    portfolio: List[Portfolio]
    url: str

class InvestmentProgress(BaseModel):
    """The progress of a investment search."""
    query: str
    results: list[InvestmentAccounts]
    done: bool



class ScholarshipProgress(BaseModel):
    """The progress of a scholarship search."""
    query: str
    results: list[Scholarship]
    done: bool

class MajorProgress(BaseModel):
    """The progress of a major search."""
    query: str
    results: list[Major]
    done: bool

class CollegePlannerAgentState(MessagesState):
    """The state of the college planner."""
    state_of_residence: str
    income: int
    filing_status: Literal["single", "married"]
    tax_bracket: Dict[int, int]
    selected_colleges: List[College]
    selected_majors: List[Major]
    selected_scholarships: List[Scholarship]
    selected_investment_accounts: List[InvestmentAccounts]
    college_progress: List[CollegeProgress] | None
    major_progress: List[MajorProgress] | None
    scholarship_progress: List[ScholarshipProgress] | None
    investment_progress: List[InvestmentProgress] | None