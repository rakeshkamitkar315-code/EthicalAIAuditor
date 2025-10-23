
"""
This package contains core modules for the Ethical AI Bias Detection project.
"""

from .preprocess import preprocess
from .fairness_check import (
    demographic_parity_difference,
    equality_of_opportunity_difference
)