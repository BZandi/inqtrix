"""Wire-contract constants shared across serving layers.

Single source for values that appear on multiple correlated wire
surfaces; duplicating them per module would let the surfaces drift
silently (Designprinzip 4).
"""

MODEL_NAME = "research-agent"
"""Public model identifier on the OpenAI-compatible surface.

Appears in three correlated places clients match on: the ``model``
field of non-streaming chat completions, the ``model`` field of every
streaming chunk, and the ``id`` of the single ``/v1/models`` entry.
"""
