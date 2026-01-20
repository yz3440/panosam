"""SAM3 Engine - DEPRECATED module location.

This module is kept for backward compatibility.
Please import from panosam.engines.sam3 instead:

    from panosam.engines.sam3 import SAM3Engine
"""

import warnings

from ..engines.sam3 import SAM3Engine

warnings.warn(
    "Importing SAM3Engine from panosam.sam.engine is deprecated. "
    "Use 'from panosam.engines.sam3 import SAM3Engine' instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SAM3Engine"]
