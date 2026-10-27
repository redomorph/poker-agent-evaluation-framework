"""Deterministic Bayesian opponent-observation model for Phase 2.

This module models only observable public-action frequencies.  It does not
select poker actions and it never consumes hidden state or agent RNG.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from typing import Any

from pokerbench.evaluation.calibration import (
    OPPORTUNITY_DEFINITION_CONTRACT_VERSION,
    BehaviorCounts,
)


BAYESIAN_OPPONENT_MODEL_VERSION = "bayesian-opponent-model-v1"
BAYESIAN_OPPONENT_PRIOR_VERSION = "bayesian-opponent-prior-v1"


class OpponentModelMetric(StrEnum):
    """Observable Bernoulli quantities tracked by the v1 learner."""

    VPIP = "vpip"
    PFR = "pfr"
    THREE_BET = "three_bet_freq"
    CBET = "cbet_freq"
    AGGRESSIVE_SHARE = "postflop_aggressive_share"
    FOLD_TO_CBET = "fold_to_cbet"


@dataclass(frozen=True)
class BetaPrior:
    """Explicit integer-shape Beta prior; no project-wide default exists."""

    alpha: int
    beta: int

    def __post_init__(self) -> None:
        for name, value in (("alpha", self.alpha), ("beta", self.beta)):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")

    def to_dict(self) -> dict[str, int]:
        return {"alpha": self.alpha, "beta": self.beta}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BetaPrior":
        _require_exact_keys(payload, {"alpha", "beta"}, "BetaPrior")
        return cls(alpha=payload["alpha"], beta=payload["beta"])


@dataclass(frozen=True)
class OpponentModelPrior:
    """Explicit prior vector required before an adaptive experiment starts."""

    vpip: BetaPrior
    pfr: BetaPrior
    three_bet: BetaPrior
    cbet: BetaPrior
    aggressive_share: BetaPrior
    fold_to_cbet: BetaPrior
    version: str = BAYESIAN_OPPONENT_PRIOR_VERSION

    def __post_init__(self) -> None:
        if self.version != BAYESIAN_OPPONENT_PRIOR_VERSION:
            raise ValueError(
                "Unsupported opponent-model prior version: "
                f"{self.version}"
            )

    def for_metric(self, metric: OpponentModelMetric) -> BetaPrior:
        mapping = {
            OpponentModelMetric.VPIP: self.vpip,
            OpponentModelMetric.PFR: self.pfr,
            OpponentModelMetric.THREE_BET: self.three_bet,
            OpponentModelMetric.CBET: self.cbet,
            OpponentModelMetric.AGGRESSIVE_SHARE: self.aggressive_share,
            OpponentModelMetric.FOLD_TO_CBET: self.fold_to_cbet,
        }
        return mapping[metric]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "vpip": self.vpip.to_dict(),
            "pfr": self.pfr.to_dict(),
            "three_bet": self.three_bet.to_dict(),
            "cbet": self.cbet.to_dict(),
            "aggressive_share": self.aggressive_share.to_dict(),
            "fold_to_cbet": self.fold_to_cbet.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OpponentModelPrior":
        expected = {
            "version",
            "vpip",
            "pfr",
            "three_bet",
            "cbet",
            "aggressive_share",
            "fold_to_cbet",
        }
        _require_exact_keys(payload, expected, "OpponentModelPrior")
        return cls(
            version=payload["version"],
            vpip=BetaPrior.from_dict(payload["vpip"]),
            pfr=BetaPrior.from_dict(payload["pfr"]),
            three_bet=BetaPrior.from_dict(payload["three_bet"]),
            cbet=BetaPrior.from_dict(payload["cbet"]),
            aggressive_share=BetaPrior.from_dict(
                payload["aggressive_share"]
            ),
            fold_to_cbet=BetaPrior.from_dict(payload["fold_to_cbet"]),
        )


@dataclass(frozen=True)
class BayesianMetricPosterior:
    """One exact Beta-Bernoulli posterior with cumulative evidence counts."""

    metric: OpponentModelMetric
    prior: BetaPrior
    events: int = 0
    opportunities: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("events", self.events),
            ("opportunities", self.opportunities),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.events > self.opportunities:
            raise ValueError("events cannot exceed opportunities")

    @property
    def alpha(self) -> int:
        return self.prior.alpha + self.events

    @property
    def beta(self) -> int:
        return self.prior.beta + self.opportunities - self.events

    @property
    def mean(self) -> Fraction:
        return Fraction(self.alpha, self.alpha + self.beta)

    @property
    def variance(self) -> Fraction:
        total = self.alpha + self.beta
        return Fraction(
            self.alpha * self.beta,
            total * total * (total + 1),
        )

    @property
    def concentration(self) -> int:
        return self.alpha + self.beta

    def updated(self, *, events: int, opportunities: int) -> "BayesianMetricPosterior":
        if not isinstance(events, int) or isinstance(events, bool):
            raise TypeError("events must be an integer")
        if not isinstance(opportunities, int) or isinstance(opportunities, bool):
            raise TypeError("opportunities must be an integer")
        if events < 0 or opportunities < 0:
            raise ValueError("evidence counts must be non-negative")
        if events > opportunities:
            raise ValueError("events cannot exceed opportunities")
        return BayesianMetricPosterior(
            metric=self.metric,
            prior=self.prior,
            events=self.events + events,
            opportunities=self.opportunities + opportunities,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric.value,
            "prior_alpha": self.prior.alpha,
            "prior_beta": self.prior.beta,
            "events": self.events,
            "opportunities": self.opportunities,
            "alpha": self.alpha,
            "beta": self.beta,
            "concentration": self.concentration,
            "mean": _fraction_dict(self.mean),
            "variance": _fraction_dict(self.variance),
        }


@dataclass(frozen=True)
class BayesianOpponentModel:
    """Immutable cumulative Bayesian model of one opponent's public actions."""

    prior: OpponentModelPrior
    hands_observed: int
    vpip: BayesianMetricPosterior
    pfr: BayesianMetricPosterior
    three_bet: BayesianMetricPosterior
    cbet: BayesianMetricPosterior
    aggressive_share: BayesianMetricPosterior
    fold_to_cbet: BayesianMetricPosterior
    model_version: str = BAYESIAN_OPPONENT_MODEL_VERSION
    opportunity_definition_version: str = OPPORTUNITY_DEFINITION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.model_version != BAYESIAN_OPPONENT_MODEL_VERSION:
            raise ValueError(f"Unsupported model version: {self.model_version}")
        if (
            self.opportunity_definition_version
            != OPPORTUNITY_DEFINITION_CONTRACT_VERSION
        ):
            raise ValueError(
                "Unsupported opportunity_definition_version: "
                f"{self.opportunity_definition_version}"
            )
        if not isinstance(self.hands_observed, int) or isinstance(
            self.hands_observed, bool
        ):
            raise TypeError("hands_observed must be an integer")
        if self.hands_observed < 0:
            raise ValueError("hands_observed must be non-negative")
        for metric in OpponentModelMetric:
            posterior = self.posterior(metric)
            if posterior.metric != metric:
                raise ValueError("posterior metric mismatch")
            if posterior.prior != self.prior.for_metric(metric):
                raise ValueError("posterior prior mismatch")
        if self.vpip.opportunities != self.hands_observed:
            raise ValueError(
                "hands_observed must equal cumulative VPIP opportunities"
            )

    @classmethod
    def from_prior(cls, prior: OpponentModelPrior) -> "BayesianOpponentModel":
        def posterior(metric: OpponentModelMetric) -> BayesianMetricPosterior:
            return BayesianMetricPosterior(
                metric=metric,
                prior=prior.for_metric(metric),
            )

        return cls(
            prior=prior,
            hands_observed=0,
            vpip=posterior(OpponentModelMetric.VPIP),
            pfr=posterior(OpponentModelMetric.PFR),
            three_bet=posterior(OpponentModelMetric.THREE_BET),
            cbet=posterior(OpponentModelMetric.CBET),
            aggressive_share=posterior(OpponentModelMetric.AGGRESSIVE_SHARE),
            fold_to_cbet=posterior(OpponentModelMetric.FOLD_TO_CBET),
        )

    def posterior(self, metric: OpponentModelMetric) -> BayesianMetricPosterior:
        mapping = {
            OpponentModelMetric.VPIP: self.vpip,
            OpponentModelMetric.PFR: self.pfr,
            OpponentModelMetric.THREE_BET: self.three_bet,
            OpponentModelMetric.CBET: self.cbet,
            OpponentModelMetric.AGGRESSIVE_SHARE: self.aggressive_share,
            OpponentModelMetric.FOLD_TO_CBET: self.fold_to_cbet,
        }
        return mapping[metric]

    @property
    def aggression_ratio_from_posterior_mean(self) -> Fraction:
        posterior = self.aggressive_share
        return Fraction(posterior.alpha, posterior.beta)

    def updated(
        self,
        counts: BehaviorCounts,
        *,
        first_hand_index: int,
    ) -> "BayesianOpponentModel":
        if not isinstance(first_hand_index, int) or isinstance(
            first_hand_index, bool
        ):
            raise TypeError("first_hand_index must be an integer")
        if first_hand_index != self.hands_observed:
            raise ValueError(
                "first_hand_index must equal the current hands_observed; "
                "duplicate or reordered evidence is not allowed"
            )
        if (
            counts.opportunity_definition_version
            != self.opportunity_definition_version
        ):
            raise ValueError(
                "opportunity_definition_version does not match the "
                "frozen model contract"
            )

        aggression_opportunities = (
            counts.postflop_aggressive_actions
            + counts.postflop_passive_actions
        )

        return BayesianOpponentModel(
            prior=self.prior,
            hands_observed=(
                self.hands_observed + counts.vpip_opportunities
            ),
            vpip=self.vpip.updated(
                events=counts.vpip_events,
                opportunities=counts.vpip_opportunities,
            ),
            pfr=self.pfr.updated(
                events=counts.pfr_events,
                opportunities=counts.pfr_opportunities,
            ),
            three_bet=self.three_bet.updated(
                events=counts.three_bet_events,
                opportunities=counts.three_bet_opportunities,
            ),
            cbet=self.cbet.updated(
                events=counts.cbet_events,
                opportunities=counts.cbet_opportunities,
            ),
            aggressive_share=self.aggressive_share.updated(
                events=counts.postflop_aggressive_actions,
                opportunities=aggression_opportunities,
            ),
            fold_to_cbet=self.fold_to_cbet.updated(
                events=counts.fold_to_cbet_events,
                opportunities=counts.fold_to_cbet_opportunities,
            ),
            model_version=self.model_version,
            opportunity_definition_version=self.opportunity_definition_version,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_version": self.model_version,
            "opportunity_definition_version": self.opportunity_definition_version,
            "hands_observed": self.hands_observed,
            "prior": self.prior.to_dict(),
            "posteriors": {
                metric.value: self.posterior(metric).to_dict()
                for metric in OpponentModelMetric
            },
            "aggression_ratio_from_posterior_mean": _fraction_dict(
                self.aggression_ratio_from_posterior_mean
            ),
        }

    @property
    def state_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BayesianOpponentModel":
        expected = {
            "model_version",
            "opportunity_definition_version",
            "hands_observed",
            "prior",
            "posteriors",
            "aggression_ratio_from_posterior_mean",
        }
        _require_exact_keys(payload, expected, "BayesianOpponentModel")
        prior = OpponentModelPrior.from_dict(payload["prior"])
        posteriors_payload = payload["posteriors"]
        if not isinstance(posteriors_payload, dict):
            raise TypeError("posteriors must be an object")

        posterior_objects: dict[OpponentModelMetric, BayesianMetricPosterior] = {}
        for metric in OpponentModelMetric:
            item = posteriors_payload.get(metric.value)
            if not isinstance(item, dict):
                raise ValueError(f"Missing posterior payload for {metric.value}")
            expected_posterior = {
                "metric",
                "prior_alpha",
                "prior_beta",
                "events",
                "opportunities",
                "alpha",
                "beta",
                "concentration",
                "mean",
                "variance",
            }
            _require_exact_keys(
                item,
                expected_posterior,
                f"posterior payload for {metric.value}",
            )
            posterior = BayesianMetricPosterior(
                metric=metric,
                prior=prior.for_metric(metric),
                events=item["events"],
                opportunities=item["opportunities"],
            )
            if item != posterior.to_dict():
                raise ValueError(
                    f"posterior payload for {metric.value} is inconsistent"
                )
            posterior_objects[metric] = posterior

        if set(posteriors_payload) != {metric.value for metric in OpponentModelMetric}:
            raise ValueError("posterior payload contains unsupported metrics")

        model = cls(
            prior=prior,
            hands_observed=payload["hands_observed"],
            vpip=posterior_objects[OpponentModelMetric.VPIP],
            pfr=posterior_objects[OpponentModelMetric.PFR],
            three_bet=posterior_objects[OpponentModelMetric.THREE_BET],
            cbet=posterior_objects[OpponentModelMetric.CBET],
            aggressive_share=posterior_objects[
                OpponentModelMetric.AGGRESSIVE_SHARE
            ],
            fold_to_cbet=posterior_objects[
                OpponentModelMetric.FOLD_TO_CBET
            ],
            model_version=payload["model_version"],
            opportunity_definition_version=payload[
                "opportunity_definition_version"
            ],
        )
        if payload["aggression_ratio_from_posterior_mean"] != _fraction_dict(
            model.aggression_ratio_from_posterior_mean
        ):
            raise ValueError("aggression ratio payload is inconsistent")
        return model


def _fraction_dict(value: Fraction) -> dict[str, int]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
    }


def _require_exact_keys(
    payload: dict[str, Any],
    expected: set[str],
    context: str,
) -> None:
    if not isinstance(payload, dict):
        raise TypeError(f"{context} must be an object")
    actual = set(payload)
    if actual != expected:
        raise ValueError(
            f"{context} has unexpected keys: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
