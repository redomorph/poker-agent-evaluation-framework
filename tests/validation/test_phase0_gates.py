# tests/validation/test_phase0_gates.py
"""
Phase-0 validation gates.

This module validates:

1. The frozen low-variance Phase-0 stochastic policy.
2. Position-controlled symmetry.
3. Random-vs-random statistical compatibility with zero.
4. The strict practical symmetry tolerance in BB/100.
5. Event-indexed RNG reproducibility.
6. Use of at least three independent master seeds.

The Phase-0 stochastic policy is a validation instrument only.
It is not a benchmark poker strategy.
"""

from __future__ import annotations

import math

import pytest

from pokerbench.agents.base import Agent
from pokerbench.agents.observation import AgentObservation
from pokerbench.engine.config import (
    Action,
    BASELINE_GAME_CONFIG,
)
from pokerbench.rng import AgentRNG
from pokerbench.runner import (
    IllegalActionPolicy,
    PairedResult,
    PairedRunner,
    StatisticalUnit,
)


# -------------------------------------------------------------------------
# Frozen Phase-0 validation configuration
# -------------------------------------------------------------------------

PHASE0_RANDOM_POLICY_VERSION = "phase0-symmetry-v1"

# Probability of folding when both FOLD and CALL are legal.
#
# This deliberately introduces a small amount of stochasticity while
# keeping the symmetry test low-variance enough for the strict
# ±0.5 BB/100 confidence-interval requirement.
PHASE0_FOLD_PROBABILITY = 0.005

# Pre-planned number of paired deals per master seed.
#
# This value is fixed before running the validation experiment.
# It must not be changed in response to the observed CI from a run.
PAIRS_PER_SEED = 5_000

# Development / sanity experiments use at least three master seeds.
MASTER_SEEDS: tuple[int, ...] = (
    42,
    12_345,
    999,
)

# Strict Phase-0 symmetry tolerance.
SYMMETRY_TOLERANCE_BB100 = 0.5

# PairedResult stores mean-like quantities in BB per paired observation.
BB100_SCALE = 100.0

# Current PairedResult normal 95% CI uses z = 1.96.
Z_95 = 1.96


# -------------------------------------------------------------------------
# Frozen Phase-0 stochastic validation agent
# -------------------------------------------------------------------------

class Phase0RandomAgent(Agent):
    """
    Low-variance stochastic agent used only for Phase-0 validation.

    Policy version:
        phase0-symmetry-v1

    The policy deliberately excludes all aggressive actions.

    When facing a wager and both FOLD and CALL are legal:

        P(FOLD) = 0.005
        P(CALL) = 0.995

    Otherwise:

        CHECK if available;
        otherwise CALL if available;
        otherwise FOLD if available.

    Exactly one RNG draw is consumed for every decision, including
    deterministic CHECK/CALL decisions. This keeps the agent decision
    index aligned with RNG consumption.
    """

    def __init__(
        self,
        name: str = "Phase0Random",
    ) -> None:
        self.name = name
        self.act_count = 0

    @staticmethod
    def _action_from_draw(
        legal_actions: list[Action],
        draw: float,
    ) -> Action:
        """
        Map one Uniform[0,1) RNG draw to the frozen Phase-0 policy.
        """
        if not 0.0 <= draw < 1.0:
            raise ValueError(
                "draw must satisfy 0.0 <= draw < 1.0"
            )

        legal = set(legal_actions)

        # Facing a wager:
        # randomise only between FOLD and CALL.
        if Action.CALL in legal:
            if (
                Action.FOLD in legal
                and draw < PHASE0_FOLD_PROBABILITY
            ):
                return Action.FOLD

            return Action.CALL

        # No wager to call:
        # check deterministically.
        if Action.CHECK in legal:
            return Action.CHECK

        # Defensive fallback for any unusual legal-action state.
        if Action.FOLD in legal:
            return Action.FOLD

        raise ValueError(
            "Phase0RandomAgent received no supported legal action"
        )

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        self.act_count += 1

        # Exactly one event-indexed random value is consumed per
        # agent decision.
        draw = rng.random()

        return self._action_from_draw(
            legal_actions=legal_actions,
            draw=draw,
        )

    @property
    def is_adaptive(self) -> bool:
        return False

    def reset(self) -> None:
        self.act_count = 0


class FoldAgent(Agent):
    """Deterministic agent used for exact structural symmetry checks."""

    def __init__(
        self,
        name: str = "Fold",
    ) -> None:
        self.name = name
        self.act_count = 0

    def act(
        self,
        observation: AgentObservation,
        legal_actions: list[Action],
        rng: AgentRNG,
    ) -> Action:
        self.act_count += 1

        return Action.FOLD

    @property
    def is_adaptive(self) -> bool:
        return False

    def reset(self) -> None:
        self.act_count = 0


# -------------------------------------------------------------------------
# Metadata
# -------------------------------------------------------------------------

def _phase0_experiment_metadata() -> dict[str, object]:
    """Return the frozen Phase-0 policy metadata."""
    return {
        "phase0_random_policy_version": (
            PHASE0_RANDOM_POLICY_VERSION
        ),
        "phase0_fold_probability": (
            PHASE0_FOLD_PROBABILITY
        ),
        "phase0_pairs_per_seed": PAIRS_PER_SEED,
    }


# -------------------------------------------------------------------------
# Statistical helpers
# -------------------------------------------------------------------------

def _require_complete_statistics(
    result: PairedResult,
) -> tuple[float, float, float, float]:
    """
    Return mean, SD, lower CI, and upper CI with Optional types narrowed.
    """
    assert result.mean_d is not None, (
        "PairedResult.mean_d must exist for a non-empty result"
    )

    assert result.std_d is not None, (
        "PairedResult.std_d must exist for multiple paired observations"
    )

    assert result.ci_low is not None, (
        "PairedResult.ci_low must exist for multiple paired observations"
    )

    assert result.ci_high is not None, (
        "PairedResult.ci_high must exist for multiple paired observations"
    )

    return (
        result.mean_d,
        result.std_d,
        result.ci_low,
        result.ci_high,
    )


def _to_bb100(
    value_bb: float,
) -> float:
    """
    Convert a mean-like quantity from BB/observation to BB/100.
    """
    return value_bb * BB100_SCALE


def _required_pairs_for_ci_half_width(
    std_bb100: float,
    target_half_width_bb100: float,
) -> int:
    """
    Return the normal-approximation sample size for a target 95% CI width.

    This is a planning / diagnostic calculation only. It must never be
    used for sequential stopping of the current experiment.
    """
    if std_bb100 < 0.0:
        raise ValueError(
            "std_bb100 must be non-negative"
        )

    if target_half_width_bb100 <= 0.0:
        raise ValueError(
            "target_half_width_bb100 must be positive"
        )

    if std_bb100 == 0.0:
        return 1

    required = (
        Z_95
        * std_bb100
        / target_half_width_bb100
    ) ** 2

    return math.ceil(required)


def _phase0_policy_sd_bound_bb100() -> float:
    """
    Return a conservative population-SD bound for Phase0RandomAgent.

    If both agents make the same BTN preflop decision, the paired
    position-controlled difference cancels exactly.

    A non-zero paired result requires a fold/call mismatch.

    Mismatch probability:

        2 * q * (1 - q)

    where q is PHASE0_FOLD_PROBABILITY.

    On such a mismatch, the absolute paired difference is at most
    1.5 BB.

    Therefore:

        E[d^2] <= 2*q*(1-q)*(1.5^2)

    and:

        SD(d) <= sqrt(4.5*q*(1-q)) BB
    """
    q = PHASE0_FOLD_PROBABILITY

    variance_bound_bb = 4.5 * q * (1.0 - q)

    return (
        math.sqrt(variance_bound_bb)
        * BB100_SCALE
    )


def _format_symmetry_diagnostics(
    *,
    seed: int,
    result: PairedResult,
    mean_bb100: float,
    std_bb100: float,
    ci_low_bb100: float,
    ci_high_bb100: float,
) -> str:
    """Return a detailed diagnostic message for symmetry failures."""
    required_pairs = _required_pairs_for_ci_half_width(
        std_bb100=std_bb100,
        target_half_width_bb100=(
            SYMMETRY_TOLERANCE_BB100
        ),
    )

    ci_half_width = (
        ci_high_bb100 - ci_low_bb100
    ) / 2.0

    return (
        f"policy={PHASE0_RANDOM_POLICY_VERSION}, "
        f"seed={seed}, "
        f"valid_pairs={result.num_valid_pairs}, "
        f"mean={mean_bb100:.3f} BB/100, "
        f"sd={std_bb100:.3f} BB/100, "
        f"95% CI=["
        f"{ci_low_bb100:.3f}, "
        f"{ci_high_bb100:.3f}] BB/100, "
        f"CI half-width={ci_half_width:.3f} BB/100, "
        f"target=["
        f"{-SYMMETRY_TOLERANCE_BB100:.3f}, "
        f"{SYMMETRY_TOLERANCE_BB100:.3f}] BB/100, "
        f"observed-SD required pairs="
        f"{required_pairs}"
    )


# -------------------------------------------------------------------------
# TDD: frozen Phase-0 policy contract
# -------------------------------------------------------------------------

def test_phase0_policy_folds_below_probability_threshold() -> None:
    legal_actions = [
        Action.FOLD,
        Action.CALL,
        Action.BET_33,
        Action.BET_100,
        Action.ALL_IN,
    ]

    action = Phase0RandomAgent._action_from_draw(
        legal_actions=legal_actions,
        draw=0.0,
    )

    assert action == Action.FOLD


def test_phase0_policy_calls_at_probability_threshold() -> None:
    legal_actions = [
        Action.FOLD,
        Action.CALL,
        Action.BET_33,
        Action.BET_100,
        Action.ALL_IN,
    ]

    action = Phase0RandomAgent._action_from_draw(
        legal_actions=legal_actions,
        draw=PHASE0_FOLD_PROBABILITY,
    )

    assert action == Action.CALL


def test_phase0_policy_never_selects_aggressive_actions() -> None:
    legal_actions = [
        Action.FOLD,
        Action.CALL,
        Action.BET_33,
        Action.BET_66,
        Action.BET_100,
        Action.BET_200,
        Action.ALL_IN,
    ]

    aggressive_actions = {
        Action.BET_33,
        Action.BET_66,
        Action.BET_100,
        Action.BET_200,
        Action.ALL_IN,
    }

    draws = (
        0.0,
        PHASE0_FOLD_PROBABILITY / 2.0,
        PHASE0_FOLD_PROBABILITY,
        0.25,
        0.50,
        0.75,
        0.999999,
    )

    for draw in draws:
        action = Phase0RandomAgent._action_from_draw(
            legal_actions=legal_actions,
            draw=draw,
        )

        assert action not in aggressive_actions


def test_phase0_policy_checks_when_check_is_available() -> None:
    legal_actions = [
        Action.CHECK,
        Action.BET_33,
        Action.BET_66,
        Action.BET_100,
        Action.BET_200,
        Action.ALL_IN,
    ]

    for draw in (
        0.0,
        0.5,
        0.999999,
    ):
        action = Phase0RandomAgent._action_from_draw(
            legal_actions=legal_actions,
            draw=draw,
        )

        assert action == Action.CHECK


def test_phase0_policy_rejects_invalid_rng_draw() -> None:
    with pytest.raises(ValueError):
        Phase0RandomAgent._action_from_draw(
            legal_actions=[
                Action.FOLD,
                Action.CALL,
            ],
            draw=1.0,
        )


def test_phase0_planned_sample_meets_design_precision_bound() -> None:
    """
    The frozen 5,000-pair budget must be at least as large as the
    normal-approximation planning requirement implied by the frozen
    policy's conservative population-SD bound.
    """
    sd_bound_bb100 = _phase0_policy_sd_bound_bb100()

    required_pairs = _required_pairs_for_ci_half_width(
        std_bb100=sd_bound_bb100,
        target_half_width_bb100=(
            SYMMETRY_TOLERANCE_BB100
        ),
    )

    assert required_pairs <= PAIRS_PER_SEED


# -------------------------------------------------------------------------
# Exact deterministic symmetry gate
# -------------------------------------------------------------------------

def test_identical_fold_agents_are_exactly_symmetric() -> None:
    """
    Two identical deterministic policies must have zero paired bias.
    """
    runner = PairedRunner(
        agent_a=FoldAgent("A"),
        agent_b=FoldAgent("B"),
        num_hands=100,
        seed=42,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=False,
        illegal_action_policy=(
            IllegalActionPolicy.EXCLUDE_RUN
        ),
    )

    result = runner.run()

    assert (
        result.statistical_unit
        == StatisticalUnit.PAIRED_DEAL
    )

    assert result.num_valid_pairs == 100
    assert result.failed_runs == []

    assert result.mean_d is not None
    assert abs(result.mean_d) < 1e-12
    assert abs(_to_bb100(result.mean_d)) < 1e-12


# -------------------------------------------------------------------------
# Random-vs-random symmetry / CI gate
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "seed",
    MASTER_SEEDS,
)
def test_symmetry_and_phase0_random_vs_random(
    seed: int,
) -> None:
    """
    Validate the frozen Phase-0 stochastic policy.

    Requirements:

    1. Every requested paired deal is valid.
    2. Statistical unit is PAIRED_DEAL.
    3. 95% CI contains zero.
    4. Complete 95% CI lies inside [-0.5,+0.5] BB/100.
    5. Frozen policy metadata is persisted.
    """
    runner = PairedRunner(
        agent_a=Phase0RandomAgent("A"),
        agent_b=Phase0RandomAgent("B"),
        num_hands=PAIRS_PER_SEED,
        seed=seed,
        game_config=BASELINE_GAME_CONFIG,
        experiment_metadata=(
            _phase0_experiment_metadata()
        ),
        adaptive=False,
        illegal_action_policy=(
            IllegalActionPolicy.EXCLUDE_RUN
        ),
    )

    result = runner.run()

    # ------------------------------------------------------------------
    # Structural validity
    # ------------------------------------------------------------------

    assert (
        result.statistical_unit
        == StatisticalUnit.PAIRED_DEAL
    )

    assert result.num_valid_pairs == PAIRS_PER_SEED, (
        f"Expected {PAIRS_PER_SEED} valid pairs, "
        f"got {result.num_valid_pairs}; "
        f"seed={seed}, failures={result.failed_runs}"
    )

    assert result.failed_runs == [], (
        f"Phase-0 random-vs-random produced failed runs; "
        f"seed={seed}, failures={result.failed_runs}"
    )

    # ------------------------------------------------------------------
    # Frozen-policy metadata
    # ------------------------------------------------------------------

    assert (
        result.experiment_metadata[
            "phase0_random_policy_version"
        ]
        == PHASE0_RANDOM_POLICY_VERSION
    )

    assert (
        result.experiment_metadata[
            "phase0_fold_probability"
        ]
        == PHASE0_FOLD_PROBABILITY
    )

    assert (
        result.experiment_metadata[
            "phase0_pairs_per_seed"
        ]
        == PAIRS_PER_SEED
    )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    (
        mean_raw,
        std_raw,
        ci_low_raw,
        ci_high_raw,
    ) = _require_complete_statistics(result)

    mean_bb100 = _to_bb100(mean_raw)
    std_bb100 = _to_bb100(std_raw)
    ci_low_bb100 = _to_bb100(ci_low_raw)
    ci_high_bb100 = _to_bb100(ci_high_raw)

    diagnostics = _format_symmetry_diagnostics(
        seed=seed,
        result=result,
        mean_bb100=mean_bb100,
        std_bb100=std_bb100,
        ci_low_bb100=ci_low_bb100,
        ci_high_bb100=ci_high_bb100,
    )

    # ------------------------------------------------------------------
    # Statistical compatibility with zero
    # ------------------------------------------------------------------

    assert (
        ci_low_bb100
        <= 0.0
        <= ci_high_bb100
    ), (
        "Phase-0 random-vs-random result is not "
        f"statistically compatible with zero. {diagnostics}"
    )

    # ------------------------------------------------------------------
    # Strict practical symmetry / precision gate
    # ------------------------------------------------------------------

    assert (
        ci_low_bb100
        >= -SYMMETRY_TOLERANCE_BB100
    ), (
        "Phase-0 random-vs-random 95% CI extends "
        f"below the symmetry tolerance. {diagnostics}"
    )

    assert (
        ci_high_bb100
        <= SYMMETRY_TOLERANCE_BB100
    ), (
        "Phase-0 random-vs-random 95% CI extends "
        f"above the symmetry tolerance. {diagnostics}"
    )


# -------------------------------------------------------------------------
# RNG / run determinism gate
# -------------------------------------------------------------------------

def test_determinism_with_phase0_random_agents() -> None:
    """
    Identical software, policy, seed, and configuration must reproduce
    exactly the same stochastic paired experiment.
    """
    seed = 999
    num_pairs = 250

    first_runner = PairedRunner(
        agent_a=Phase0RandomAgent("A"),
        agent_b=Phase0RandomAgent("B"),
        num_hands=num_pairs,
        seed=seed,
        game_config=BASELINE_GAME_CONFIG,
        experiment_metadata=(
            _phase0_experiment_metadata()
        ),
        adaptive=False,
        illegal_action_policy=(
            IllegalActionPolicy.EXCLUDE_RUN
        ),
    )

    second_runner = PairedRunner(
        agent_a=Phase0RandomAgent("A"),
        agent_b=Phase0RandomAgent("B"),
        num_hands=num_pairs,
        seed=seed,
        game_config=BASELINE_GAME_CONFIG,
        experiment_metadata=(
            _phase0_experiment_metadata()
        ),
        adaptive=False,
        illegal_action_policy=(
            IllegalActionPolicy.EXCLUDE_RUN
        ),
    )

    result_first = first_runner.run()
    result_second = second_runner.run()

    assert (
        result_first.paired_differences
        == result_second.paired_differences
    )

    assert result_first.seed == result_second.seed

    assert (
        result_first.statistical_unit
        == result_second.statistical_unit
    )

    assert (
        result_first.num_hands
        == result_second.num_hands
    )

    assert (
        result_first.num_valid_pairs
        == result_second.num_valid_pairs
    )

    assert (
        result_first.failed_runs
        == result_second.failed_runs
    )

    assert (
        result_first.experiment_metadata
        == result_second.experiment_metadata
    )

    assert result_first.mean_d == result_second.mean_d
    assert result_first.std_d == result_second.std_d
    assert result_first.ci_low == result_second.ci_low
    assert result_first.ci_high == result_second.ci_high


# -------------------------------------------------------------------------
# Seed-set contract
# -------------------------------------------------------------------------

def test_phase0_uses_at_least_three_master_seeds() -> None:
    """Development / sanity validation must use at least three seeds."""
    assert len(MASTER_SEEDS) >= 3
    assert len(set(MASTER_SEEDS)) == len(MASTER_SEEDS)