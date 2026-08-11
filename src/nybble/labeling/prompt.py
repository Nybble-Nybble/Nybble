"""Prompts used by the Gemini-first frame labeling pipeline."""

import json
from collections.abc import Sequence

from nybble.labeling.schema import FrameLabel

FRAME_CAPTION_PROMPT_VERSION = "nybble-frame-caption-v1"
GROUPING_PROMPT_VERSION = "nybble-semantic-grouping-v1"

# This preserves the important contract of the original root prompt: one image is
# described independently, in the present tense, with no inferred future action.
FRAME_CAPTION_PROMPT = """\
This is a first-person view from smart glasses worn by a person. Describe only what
is visible in this one frame, in one paragraph and in the present tense. Refer to the
person as "the wearer".

Start with the wearer's current action. If a hand or arm is visible, say exactly what
it is doing and which visible object it touches or holds. If no hand or arm is
visible, say that the wearer's hands are out of view and describe what they are
looking at. Describe every other visible person, including their position, facing
direction, action, and whether they engage with the wearer. Then name relevant
objects, the setting, what is within reach, the view direction, and visible motion
blur.

Report only visible evidence. Do not infer identity, mood, intent, a prior action, or
what happens next. Do not combine this frame with any other frame. Do not use a
preamble or bullets. Do not repeat a fact. Stay under 240 words.\
"""


def build_grouping_prompt(frames: Sequence[FrameLabel]) -> str:
    """Build a text-only grouping prompt from already independent captions.

    The indices in the model response are local, zero-based indices into ``frames``
    and both endpoints are inclusive. A chunk-wide dense context is deliberately a
    top-level field. Callers must not attach it to an event before the chunk's
    ``available_at`` time.
    """

    observations = [
        {
            "index": local_index,
            "captured_at": frame.captured_at.isoformat(),
            "caption": frame.caption,
        }
        for local_index, frame in enumerate(frames)
    ]
    return """\
Group these ordered, independently captioned egocentric frames into semantic action
spans. Return JSON matching the supplied schema.

Rules:
- Indices are zero-based into the observations below.
- start_index and end_index are inclusive.
- The first span must start at 0 and the last must end at the final observation.
- Spans must be ordered, contiguous, nonoverlapping, and cover every observation
  exactly once.
- Merge adjacent frames only when they show the same continuing user action.
- Start a new span when the wearer changes action, object of interaction, person of
  interaction, or meaningful setting.
- Each span caption must be a concise present-tense action grounded only in its
  covered observations. Do not predict the next action.
- dense_context may summarize the chunk as a whole, but must contain no prediction.

Observations:
""" + json.dumps(observations, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "FRAME_CAPTION_PROMPT",
    "FRAME_CAPTION_PROMPT_VERSION",
    "GROUPING_PROMPT_VERSION",
    "build_grouping_prompt",
]
