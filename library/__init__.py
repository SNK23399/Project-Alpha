"""
Signal libraries for ETF analysis and portfolio optimization.

Modules:
- signal_bases: Core signal computation (293 signals)
- signal_filters: Filtering and smoothing transforms (25 filters)
- signal_indicators: Cross-sectional feature transformations
"""

from .signal_bases import compute_signal_bases_generator, count_total_signals
from .signal_filters import (
    compute_filtered_signals_optimized,
    get_available_filters,
    GPU_AVAILABLE
)
from .signal_indicators import compute_all_indicators_fast, FEATURE_NAMES

__all__ = [
    'compute_signal_bases_generator',
    'count_total_signals',
    'compute_filtered_signals_optimized',
    'get_available_filters',
    'GPU_AVAILABLE',
    'compute_all_indicators_fast',
    'FEATURE_NAMES',
]
