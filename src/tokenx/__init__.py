"""tokenx – LLM cost & latency metering."""

from importlib.metadata import version as _v, PackageNotFoundError
from .cost_calc import OpenAICostCalculator  # noqa: F401
from .metrics import measure_cost, measure_latency  # noqa: F401

try:
    __version__ = _v(__name__)
except PackageNotFoundError:  # local tree
    __version__ = "0.0.dev0"
