"""Viability pressure, needs, affect, and social-pressure primitives.

Extracted from the integrated reference engine; behavior-preserving move.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Callable, Deque, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import jit, lax, random, vmap

def _clamp01(value: float, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = default
    if out != out or out in (float("inf"), float("-inf")):
        out = default
    return max(0.0, min(1.0, out))

@dataclass
class ViabilityState:
    energy: float = 1.0
    bodily_integrity: float = 1.0
    memory_integrity: float = 1.0
    prediction_coherence: float = 1.0
    social_stability: float = 1.0
    attachment_integrity: float = 1.0
    environmental_safety: float = 1.0

@dataclass
class NeedPressure:
    metabolic: float = 0.0
    repair: float = 0.0
    mnemonic: float = 0.0
    epistemic: float = 0.0
    social: float = 0.0
    attachment: float = 0.0
    safety: float = 0.0

    @classmethod
    def from_viability(cls, viability: ViabilityState) -> "NeedPressure":
        return cls(
            metabolic=1.0 - _clamp01(viability.energy, 1.0),
            repair=1.0 - _clamp01(viability.bodily_integrity, 1.0),
            mnemonic=1.0 - _clamp01(viability.memory_integrity, 1.0),
            epistemic=1.0 - _clamp01(viability.prediction_coherence, 1.0),
            social=1.0 - _clamp01(viability.social_stability, 1.0),
            attachment=1.0 - _clamp01(viability.attachment_integrity, 1.0),
            safety=1.0 - _clamp01(viability.environmental_safety, 1.0),
        )

    @property
    def total(self) -> float:
        return _clamp01(max(
            self.metabolic, self.repair, self.mnemonic, self.epistemic,
            self.social, self.attachment, self.safety))

    @property
    def dominant(self) -> str:
        values = {
            "metabolic": self.metabolic,
            "repair": self.repair,
            "mnemonic": self.mnemonic,
            "epistemic": self.epistemic,
            "social": self.social,
            "attachment": self.attachment,
            "safety": self.safety,
        }
        return max(values, key=values.get)

    def cognitive_modulators(self) -> dict:
        return {
            "risk_tolerance": _clamp01(1.0 - 0.65 * self.safety - 0.25 * self.total),
            "learning_gate": _clamp01(1.0 - 0.35 * self.metabolic - 0.25 * self.repair),
            "attention_urgency": self.total,
        }

@dataclass
class PressureAffectField:
    stability: float = 0.75
    safety: float = 0.75
    threat_salience: float = 0.0
    attachment_security: float = 0.5
    cognitive_load: float = 0.0
    prediction_coherence: float = 0.75

    def update(self, *, prediction_error: float, uncertainty: float, threat: float,
               attachment_delta: float = 0.0, social_support: float = 0.5,
               control_feedback: float = 0.5) -> "PressureAffectField":
        prediction_error = _clamp01(prediction_error)
        uncertainty = _clamp01(uncertainty)
        threat = _clamp01(threat)
        social_support = _clamp01(social_support, 0.5)
        control_feedback = _clamp01(control_feedback, 0.5)
        self.threat_salience = _clamp01(0.80 * self.threat_salience + 0.20 * threat)
        self.safety = _clamp01(1.0 - self.threat_salience)
        self.prediction_coherence = _clamp01(1.0 - prediction_error)
        self.attachment_security = _clamp01(self.attachment_security + 0.10 * attachment_delta)
        self.cognitive_load = _clamp01(0.55 * uncertainty + 0.30 * threat + 0.15 * prediction_error)
        self.stability = _clamp01(
            0.35 * self.prediction_coherence
            + 0.25 * self.safety
            + 0.20 * social_support
            + 0.20 * control_feedback
            - 0.15 * self.cognitive_load)
        return self

@dataclass
class OtherMindModel:
    agent_id: str
    trust: float = 0.5
    respect: float = 0.5
    uncertainty: float = 0.5
    beliefs: dict = field(default_factory=dict)

    def update_belief(self, claim: str, confidence: float, source_trust: float) -> None:
        prior = float(self.beliefs.get(claim, 0.5))
        weight = 0.15 + 0.70 * _clamp01(source_trust, 0.5)
        self.beliefs[claim] = _clamp01(prior * (1.0 - weight) + _clamp01(confidence) * weight)
        self.uncertainty = _clamp01(0.90 * self.uncertainty + 0.10 * (1.0 - source_trust))

@dataclass
class CommunicationIntent:
    target_agent: str
    intended_effect: str
    belief_to_modify: str
    confidence: float
    emotional_weight: float
    social_risk: float
    urgency: float = 0.0

@dataclass
class MessageInterpretation:
    accepted_confidence: float
    suspicion_delta: float

def interpret_intent(intent: CommunicationIntent, listener: OtherMindModel,
                     affect: PressureAffectField) -> MessageInterpretation:
    receptivity = _clamp01(
        0.45 * listener.trust
        + 0.25 * listener.respect
        + 0.20 * (1.0 - listener.uncertainty)
        - 0.20 * intent.social_risk
        - 0.10 * affect.threat_salience)
    accepted = _clamp01(intent.confidence * receptivity)
    suspicion = _clamp01(intent.social_risk * (1.0 - listener.trust))
    return MessageInterpretation(accepted_confidence=accepted, suspicion_delta=suspicion)

@dataclass
class SocialMemory:
    events: list = field(default_factory=list)
    max_events: int = 1024

    def remember(self, *, agent_id: str, kind: str, salience: float,
                 valence: float, claim: str | None = None) -> None:
        self.events.append({
            "agent_id": str(agent_id),
            "kind": str(kind),
            "salience": _clamp01(salience),
            "valence": max(-1.0, min(1.0, float(valence))),
            "claim": claim,
        })
        if len(self.events) > self.max_events:
            del self.events[:len(self.events) - self.max_events]

@dataclass
class FutureOutcome:
    action: str
    stability_delta: float
    social_delta: float
    expected_risk: float
    confidence: float
    viability_delta: float = 0.0
    cognitive_cost: float = 0.04

    @property
    def value(self) -> float:
        return (
            0.45 * self.stability_delta
            + 0.20 * self.social_delta
            + 0.25 * self.viability_delta
            + 0.15 * self.confidence
            - 0.35 * self.expected_risk
            - 0.20 * self.cognitive_cost
        )

def simulate_future(*, action: str, needs: NeedPressure, affect: PressureAffectField,
                    listener: OtherMindModel | None = None,
                    intent: CommunicationIntent | None = None) -> FutureOutcome:
    risk = _clamp01(max(needs.safety, needs.repair, affect.threat_salience))
    stability = 0.20 * (1.0 - needs.total) - 0.15 * affect.cognitive_load
    social = 0.0
    confidence = _clamp01(affect.prediction_coherence * (1.0 - affect.cognitive_load))
    if action in {"seek_food", "rest", "repair"}:
        stability += 0.30 * max(needs.metabolic, needs.repair)
    elif action in {"verify", "investigate"}:
        stability += 0.25 * needs.epistemic
        risk = _clamp01(risk + 0.08)
    if intent is not None and listener is not None:
        interpretation = interpret_intent(intent, listener, affect)
        social += 0.35 * interpretation.accepted_confidence - 0.10 * interpretation.suspicion_delta
        confidence = _clamp01(0.5 * confidence + 0.5 * interpretation.accepted_confidence)
    return FutureOutcome(
        action=str(action),
        stability_delta=max(-1.0, min(1.0, stability)),
        social_delta=max(-1.0, min(1.0, social)),
        expected_risk=risk,
        confidence=confidence,
        viability_delta=max(-1.0, min(1.0, stability - 0.25 * risk)),
    )
