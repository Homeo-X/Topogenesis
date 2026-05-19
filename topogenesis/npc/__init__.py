"""NPC cognition primitives for RPG-style Topogenesis agents."""

from .affect import AffectField
from .communication import CommunicationIntent, MessageInterpretation, interpret_intent
from .imagination import FutureOutcome, simulate_future
from .needs import NeedPressure, ViabilityState
from .social import Attachment, OtherMindModel, SocialMemory

__all__ = [
    "AffectField",
    "Attachment",
    "CommunicationIntent",
    "FutureOutcome",
    "MessageInterpretation",
    "NeedPressure",
    "OtherMindModel",
    "SocialMemory",
    "ViabilityState",
    "interpret_intent",
    "simulate_future",
]
