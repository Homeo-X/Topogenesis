from __future__ import annotations

from dataclasses import dataclass


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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
        prediction_error = _clamp01(prediction_error)
        uncertainty = _clamp01(uncertainty)
        threat = _clamp01(threat)
        social_support = _clamp01(social_support)
        control_feedback = max(-1.0, min(1.0, float(control_feedback)))

        self.prediction_coherence = _clamp01(
            self.prediction_coherence * (1.0 - decay)
            + decay * (1.0 - 0.65 * prediction_error - 0.35 * uncertainty)
        )
        self.threat_salience = _clamp01(
            self.threat_salience * (1.0 - decay)
            + decay * max(threat, uncertainty * 0.5)
        )
        self.control = _clamp01(
            self.control * (1.0 - decay)
            + decay * (0.55 + 0.35 * control_feedback - 0.45 * uncertainty)
        )
        self.social_belonging = _clamp01(
            self.social_belonging * (1.0 - decay)
            + decay * (0.45 + 0.55 * social_support)
        )
        if attachment_delta < 0.0:
            self.attachment_loss = _clamp01(
                self.attachment_loss + min(1.0, abs(attachment_delta)))
        else:
            self.attachment_loss = _clamp01(
                self.attachment_loss * (1.0 - decay) - 0.2 * attachment_delta)
        self.attachment_security = _clamp01(
            self.attachment_security * (1.0 - decay)
            + decay * (self.social_belonging - self.attachment_loss)
        )
        self.cognitive_load = _clamp01(
            self.cognitive_load * (1.0 - decay)
            + decay * (0.4 * uncertainty + 0.4 * prediction_error + 0.2 * threat)
        )
        self.novelty = _clamp01(
            self.novelty * (1.0 - decay) + decay * uncertainty)
        self.safety = _clamp01(1.0 - 0.55 * self.threat_salience
                               - 0.25 * uncertainty
                               - 0.20 * self.attachment_loss)
        self.stability = _clamp01(
            0.35 * self.safety
            + 0.25 * self.control
            + 0.25 * self.prediction_coherence
            + 0.15 * self.social_belonging
        )
        return self

    @property
    def risk_aversion(self) -> float:
        return _clamp01(0.30 + 0.45 * self.threat_salience
                        + 0.25 * (1.0 - self.control))

    @property
    def rumor_susceptibility(self) -> float:
        return _clamp01(0.20 + 0.45 * (1.0 - self.prediction_coherence)
                        + 0.25 * self.threat_salience
                        + 0.10 * self.social_belonging)

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
