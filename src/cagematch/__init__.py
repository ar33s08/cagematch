"""cagematch — evidence-locked arena for agent harnesses.

Run the same task suite across providers and harness variants, score cost,
latency, tool accuracy and failure modes, and prove every score by replaying
the persisted, hash-chained trace.
"""

__version__ = "0.1.0"

from cagematch.provider import Completion, OpenAICompatProvider, ProviderError, ToolCall
from cagematch.trace import RunStore, TraceTampered

__all__ = [
    "Completion",
    "OpenAICompatProvider",
    "ProviderError",
    "RunStore",
    "ToolCall",
    "TraceTampered",
    "__version__",
]
