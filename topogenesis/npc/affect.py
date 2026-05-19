from __future__ import annotations

from dataclasses import dataclass

from ._math import clamp, clamp01


@dataclass
class AffectField:
    """Continuous internal pressure field, not discrete emotion labels."""

    stability: float = 0.75
    safety: float = 0.75
    control: float = 0.65
    novelty: float = 0.25
    cognitive_load: float = 0.20
    prediction_coherence: float = 0.75
    social_belonging: float = 0.60
    attachment_security: float = 0.60
    threat_salience: float = 0.10
    attachment_loss: float = 0.0

    def __post_init__(self) -> None:
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, clamp01(getattr(self, field_name)))

    def update(
        self,
        *,
        prediction_error: float,
        uncertainty: float,
        threat: float = 0.0,
        attachment_delta: float = 0.0,
        social_support: float = 0.0,
        control_feedback: float = 0.0,
        decay: float = 0.08,
    ) -> "AffectField":
        """Advance affect as dynamical constraints under pressure."""
        prediction_error = clamp01(prediction_error)
        uncertainty = clamp01(uncertainty)
        threat = clamp01(threat)
        social_support = clamp01(social_support)
        control_feedback = clamp(control_feedback, -1.0, 1.0)
        decay = clamp(decay, 0.0, 1.0, default=0.08)

        self.prediction_coherence = clamp01(
            self.prediction_coherence * (1.0 - decay)
            + decay * (1.0 - 0.65 * prediction_error - 0.35 * uncertainty)
        )
        self.threat_salience = clamp01(
            self.threat_salience * (1.0 - decay)
            + decay * max(threat, uncertainty * 0.5)
        )
        self.control = clamp01(
            self.control * (1.0 - decay)
            + decay * (0.55 + 0.35 * control_feedback - 0.45 * uncertainty)
        )
        self.social_belonging = clamp01(
            self.social_belonging * (1.0 - decay)
            + decay * (0.45 + 0.55 * social_support)
        )
        if attachment_delta < 0.0:
            self.attachment_loss = clamp01(
                self.attachment_loss + min(1.0, abs(attachment_delta)))
        else:
            self.attachment_loss = clamp01(
                self.attachment_loss * (1.0 - decay) - 0.2 * attachment_delta)
        self.attachment_security = clamp01(
            self.attachment_security * (1.0 - decay)
            + decay * (self.social_belonging - self.attachment_loss)
        )
        self.cognitive_load = clamp01(
            self.cognitive_load * (1.0 - decay)
            + decay * (0.4 * uncertainty + 0.4 * prediction_error + 0.2 * threat)
        )
        self.novelty = clamp01(
            self.novelty * (1.0 - decay) + decay * uncertainty)
        self.safety = clamp01(1.0 - 0.55 * self.threat_salience
                              - 0.25 * uncertainty
                              - 0.20 * self.attachment_loss)
        self.stability = clamp01(
            0.35 * self.safety
            + 0.25 * self.control
            + 0.25 * self.prediction_coherence
            + 0.15 * self.social_belonging
        )
        return self

    @property
    def risk_aversion(self) -> float:
        return clamp01(0.30 + 0.45 * self.threat_salience
                       + 0.25 * (1.0 - self.control))

    @property
    def rumor_susceptibility(self) -> float:
        return clamp01(0.20 + 0.45 * (1.0 - self.prediction_coherence)
                       + 0.25 * self.threat_salience
                       + 0.10 * self.social_belonging)

    def copy(self) -> "AffectField":
        return AffectField(**self.as_dict())

    def as_dict(self) -> dict[str, float]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

    def as_vector(self) -> tuple[float, ...]:
        return (
            self.stability,
            self.safety,
            self.control,
            self.novelty,
            self.cognitive_load,
            self.prediction_coherence,
            self.social_belonging,
            self.attachment_security,
            self.threat_salience,
            self.attachment_loss,
        )
