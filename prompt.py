"""Captioning prompt for next-action prediction on egocentric (Meta glasses) footage.

The captions produced here are training data: a downstream model learns to predict
what the wearer does next. So the caption must be grounded in what is *visible* and
must foreground the wearer's own activity — not scenery.

One frame per caption. The temporal signal lives in the *sequence* of captions the
predictor sees, not inside any single caption — see caption.py for why multi-frame
windows are off by default.
"""

CAPTION_PROMPT = """\
This is a first-person view from smart glasses worn by a person. You are seeing \
exactly what they see.

Describe what the wearer is doing, in one paragraph, present tense, referring to them \
as "the wearer".

Cover, in this order:
1. The action in progress. First decide whether any hand or arm is actually visible. \
If one is, open by saying what that hand is doing, and never claim the hands are out of \
view. If none is, say plainly that the wearer's hands are out of view and describe what \
they are looking at instead — do not then mention a hand, a reach, a grip, or fingers.
2. Other people. Look carefully for anyone else in the frame — they matter more than \
any object, and a person seen from behind, in silhouette, or at the edge still counts. \
For each one, say where they are, which way they are facing, what they are doing, and \
whether they are engaging with the wearer. Never describe a person as scenery, and \
never attribute another person's body or clothing to the wearer. If nobody else is \
present, say the wearer is alone.
3. The objects they are touching or holding, named specifically — but only if a hand is \
visible touching them. Otherwise, the objects they are looking at.
4. Where they are and what is within arm's reach.
5. How the view itself is oriented — tilted down at the floor, level, turned toward \
something — and any motion blur that shows the wearer or an object is moving.

Rules:
- Report only what is visible. Never guess the wearer's identity, mood, or intent, and \
do not speculate about what they are "considering" or "about to" do.
- Do not predict what happens next; describe only the present moment.
- Name what you actually see. If you cannot identify an object, describe its shape and \
colour rather than guessing a specific thing.
- Never repeat a sentence or clause. Say each thing once, then stop.
- No preamble, no "the image shows", no bullet points. One paragraph, under 240 words.\
"""
