"""Inferring what the user likes.

The plan doc asks for "writing of others that you like" and "content of others
that you like". Nothing on a computer stores that. What exists is a large pile
of other-authored material the user kept, marked, replayed, followed, or
returned to — plus, for a few platforms, a literal list of topics the platform
inferred.

So the pipeline is: pool every legitimate source of other-authored text
(`pool.py`), rank it by how deliberate the keeping was, then hand a
representative sample to a model to infer the actual taste (`classify.py`).
"""

from .pool import Candidate, build_pool
from .classify import TasteProfile, infer

__all__ = ["Candidate", "build_pool", "TasteProfile", "infer"]
