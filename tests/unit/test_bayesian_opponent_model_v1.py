from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import pytest

from pokerbench.agents.opponent_model import (
    BAYESIAN_OPPONENT_MODEL_VERSION,
    BAYESIAN_OPPONENT_PRIOR_VERSION,
    BetaPrior,
    BayesianOpponentModel,
    OpponentModelMetric,
    OpponentModelPrior,
)
from pokerbench.evaluation.calibration import (
    OPPORTUNITY_DEFINITION_CONTRACT_VERSION,
    BehaviorCounts,
)


def _prior() -> OpponentModelPrior:
    return OpponentModelPrior(
        vpip=BetaPrior(1, 1),
        pfr=BetaPrior(2, 3),
        three_bet=BetaPrior(3, 4),
        cbet=BetaPrior(4, 5),
        aggressive_share=BetaPrior(5, 6),
        fold_to_cbet=BetaPrior(6, 7),
    )


def _counts() -> BehaviorCounts:
    return BehaviorCounts(
        vpip_opportunities=4,
        vpip_events=3,
        pfr_opportunities=3,
        pfr_events=2,
        three_bet_opportunities=2,
        three_bet_events=1,
        cbet_opportunities=2,
        cbet_events=1,
        postflop_aggressive_actions=3,
        postflop_passive_actions=2,
        fold_to_cbet_opportunities=2,
        fold_to_cbet_events=1,
        opportunity_definition_version=(
            OPPORTUNITY_DEFINITION_CONTRACT_VERSION
        ),
    )


def test_beta_prior_requires_positive_integer_parameters() -> None:
    with pytest.raises(ValueError):
        BetaPrior(0, 1)
    with pytest.raises(ValueError):
        BetaPrior(1, 0)
    with pytest.raises(TypeError):
        BetaPrior(True, 1)
    with pytest.raises(TypeError):
        BetaPrior(1.5, 2)  # type: ignore[arg-type]


def test_versions_are_frozen() -> None:
    assert BAYESIAN_OPPONENT_MODEL_VERSION == "bayesian-opponent-model-v1"
    assert BAYESIAN_OPPONENT_PRIOR_VERSION == "bayesian-opponent-prior-v1"
    assert OPPORTUNITY_DEFINITION_CONTRACT_VERSION == "calibration-opportunities-v2"


def test_initial_model_is_exactly_the_explicit_prior() -> None:
    model = BayesianOpponentModel.from_prior(_prior())

    assert model.hands_observed == 0
    assert model.posterior(OpponentModelMetric.VPIP).alpha == 1
    assert model.posterior(OpponentModelMetric.VPIP).beta == 1
    assert model.posterior(OpponentModelMetric.PFR).alpha == 2
    assert model.posterior(OpponentModelMetric.PFR).beta == 3
    assert model.posterior(OpponentModelMetric.AGGRESSIVE_SHARE).alpha == 5
    assert model.posterior(OpponentModelMetric.AGGRESSIVE_SHARE).beta == 6


def test_update_uses_beta_bernoulli_conjugacy_for_all_metrics() -> None:
    updated = BayesianOpponentModel.from_prior(_prior()).updated(
        _counts(),
        first_hand_index=0,
    )

    assert updated.hands_observed == 4

    vpip = updated.posterior(OpponentModelMetric.VPIP)
    assert (vpip.alpha, vpip.beta) == (4, 2)
    assert vpip.events == 3
    assert vpip.opportunities == 4
    assert vpip.mean == Fraction(2, 3)
    assert vpip.variance == Fraction(2, 63)

    pfr = updated.posterior(OpponentModelMetric.PFR)
    assert (pfr.alpha, pfr.beta) == (4, 4)

    three_bet = updated.posterior(OpponentModelMetric.THREE_BET)
    assert (three_bet.alpha, three_bet.beta) == (4, 5)

    cbet = updated.posterior(OpponentModelMetric.CBET)
    assert (cbet.alpha, cbet.beta) == (5, 6)

    fold = updated.posterior(OpponentModelMetric.FOLD_TO_CBET)
    assert (fold.alpha, fold.beta) == (7, 8)


def test_aggression_is_modelled_as_bernoulli_aggressive_share() -> None:
    updated = BayesianOpponentModel.from_prior(_prior()).updated(
        _counts(),
        first_hand_index=0,
    )

    aggression = updated.posterior(
        OpponentModelMetric.AGGRESSIVE_SHARE
    )
    assert aggression.events == 3
    assert aggression.opportunities == 5
    assert (aggression.alpha, aggression.beta) == (8, 8)
    assert aggression.mean == Fraction(1, 2)
    assert updated.aggression_ratio_from_posterior_mean == Fraction(1, 1)


def test_zero_metric_opportunities_leave_that_posterior_unchanged() -> None:
    model = BayesianOpponentModel.from_prior(_prior())
    counts = BehaviorCounts(
        vpip_opportunities=2,
        vpip_events=1,
        opportunity_definition_version=(
            OPPORTUNITY_DEFINITION_CONTRACT_VERSION
        ),
    )
    updated = model.updated(counts, first_hand_index=0)

    assert updated.hands_observed == 2
    assert updated.posterior(OpponentModelMetric.VPIP).alpha == 2
    assert updated.posterior(OpponentModelMetric.VPIP).beta == 2
    assert updated.posterior(OpponentModelMetric.PFR) == model.posterior(
        OpponentModelMetric.PFR
    )
    assert updated.posterior(
        OpponentModelMetric.AGGRESSIVE_SHARE
    ) == model.posterior(OpponentModelMetric.AGGRESSIVE_SHARE)


def test_sequential_and_batch_updates_are_identical() -> None:
    first = BehaviorCounts(
        vpip_opportunities=2,
        vpip_events=2,
        pfr_opportunities=1,
        pfr_events=1,
        postflop_aggressive_actions=1,
        postflop_passive_actions=1,
        opportunity_definition_version=(
            OPPORTUNITY_DEFINITION_CONTRACT_VERSION
        ),
    )
    second = BehaviorCounts(
        vpip_opportunities=2,
        vpip_events=1,
        pfr_opportunities=2,
        pfr_events=1,
        three_bet_opportunities=2,
        three_bet_events=1,
        cbet_opportunities=2,
        cbet_events=1,
        postflop_aggressive_actions=2,
        postflop_passive_actions=1,
        fold_to_cbet_opportunities=2,
        fold_to_cbet_events=1,
        opportunity_definition_version=(
            OPPORTUNITY_DEFINITION_CONTRACT_VERSION
        ),
    )

    sequential = BayesianOpponentModel.from_prior(_prior())
    sequential = sequential.updated(first, first_hand_index=0)
    sequential = sequential.updated(second, first_hand_index=2)

    batch = BayesianOpponentModel.from_prior(_prior()).updated(
        _counts(),
        first_hand_index=0,
    )

    assert sequential == batch
    assert sequential.state_hash == batch.state_hash


def test_first_hand_index_prevents_duplicate_or_reordered_evidence() -> None:
    model = BayesianOpponentModel.from_prior(_prior())
    updated = model.updated(_counts(), first_hand_index=0)

    with pytest.raises(ValueError, match="first_hand_index"):
        updated.updated(_counts(), first_hand_index=0)

    with pytest.raises(ValueError, match="first_hand_index"):
        model.updated(_counts(), first_hand_index=1)


def test_old_opportunity_contract_is_rejected() -> None:
    counts = replace(
        _counts(),
        opportunity_definition_version="calibration-opportunities-v1",
    )

    with pytest.raises(ValueError, match="opportunity_definition_version"):
        BayesianOpponentModel.from_prior(_prior()).updated(
            counts,
            first_hand_index=0,
        )


def test_posterior_snapshot_is_integer_only_and_hash_stable() -> None:
    updated = BayesianOpponentModel.from_prior(_prior()).updated(
        _counts(),
        first_hand_index=0,
    )

    payload = updated.to_dict()
    assert payload["model_version"] == "bayesian-opponent-model-v1"
    assert payload["prior"]["version"] == "bayesian-opponent-prior-v1"
    assert payload["opportunity_definition_version"] == (
        "calibration-opportunities-v2"
    )
    assert payload["hands_observed"] == 4
    assert payload["posteriors"]["vpip"]["mean"] == {
        "numerator": 2,
        "denominator": 3,
    }
    assert payload["posteriors"]["vpip"]["variance"] == {
        "numerator": 2,
        "denominator": 63,
    }
    assert len(updated.state_hash) == 64
    assert updated.state_hash == BayesianOpponentModel.from_dict(payload).state_hash


def test_from_dict_rejects_tampered_derived_posterior() -> None:
    updated = BayesianOpponentModel.from_prior(_prior()).updated(
        _counts(),
        first_hand_index=0,
    )
    payload = updated.to_dict()
    payload["posteriors"]["vpip"]["alpha"] += 1

    with pytest.raises(ValueError, match="posterior payload"):
        BayesianOpponentModel.from_dict(payload)
