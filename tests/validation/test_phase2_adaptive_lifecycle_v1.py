from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from pokerbench.agents.adaptive_lifecycle import (
    ADAPTIVE_HAND_STRENGTH_AGENT_VERSION,
    ADAPTIVE_LIFECYCLE_PROTOCOL_VERSION,
    AdaptiveHandStrengthAgent,
    replay_adaptive_learning_for_bot,
)
from pokerbench.agents.base import Agent
from pokerbench.agents.budget import AgentExecutionBudget, TimeoutPolicy
from pokerbench.agents.opponent_model import BayesianOpponentModel
from pokerbench.agents.opponent_range import load_preflop_strength_reference
from pokerbench.engine.config import Action
from pokerbench.engine.config import BASELINE_GAME_CONFIG
from pokerbench.evaluation.calibration import BehaviorCounts
from pokerbench.experiments.adaptive_lifecycle_protocol import (
    ADAPTIVE_LIFECYCLE_PROTOCOL_HASH,
    load_adaptive_lifecycle_protocol,
)
from pokerbench.replay.archive import (
    HandTermination,
    MatchRecord,
    RecordedAdaptiveLearningEvent,
    RecordedHand,
)
from pokerbench.rng import AgentRNG
from pokerbench.runner import PairedRunner, StatisticalUnit
import pokerbench.runner as runner_module

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "configs" / "experiments" / "phase2-adaptive-lifecycle-v1.json"
REFERENCE_PATH = ROOT / "configs" / "experiments" / "preflop-range-strength-reference-v1.json"


class _CallCheckAgent(Agent):
    def act(self, observation, legal_actions, rng):  # type: ignore[no-untyped-def]
        del observation, rng
        if Action.CHECK in legal_actions:
            return Action.CHECK
        if Action.CALL in legal_actions:
            return Action.CALL
        return Action.FOLD


class _LifecycleProbeAgent(_CallCheckAgent):
    def __init__(self) -> None:
        self.reset_calls = 0
        self.transitions: list[RecordedAdaptiveLearningEvent] = []
        self._model = AdaptiveHandStrengthAgent.initial_model()

    @property
    def is_adaptive(self) -> bool:
        return True

    @property
    def model(self) -> BayesianOpponentModel:
        return self._model

    def reset(self) -> None:
        self.reset_calls += 1
        self._model = AdaptiveHandStrengthAgent.initial_model()

    def observe_opponent_hand(
        self,
        counts: BehaviorCounts,
        *,
        learner_bot_id: int,
        opponent_bot_id: int,
        trajectory_hand_index: int,
    ) -> RecordedAdaptiveLearningEvent:
        before = self._model
        after = before.updated(counts, first_hand_index=before.hands_observed)
        event = RecordedAdaptiveLearningEvent.from_transition(
            learner_bot_id=learner_bot_id,
            opponent_bot_id=opponent_bot_id,
            trajectory_hand_index=trajectory_hand_index,
            before=before,
            evidence=counts,
            after=after,
        )
        self._model = after
        self.transitions.append(event)
        return event


class _TimingProbeAgent(_LifecycleProbeAgent):
    def __init__(self) -> None:
        super().__init__()
        self.timeline: list[tuple[str, int]] = []

    def reset(self) -> None:
        super().reset()
        self.timeline.append(("reset", self.model.hands_observed))

    def act(self, observation, legal_actions, rng):  # type: ignore[no-untyped-def]
        self.timeline.append(("act", self.model.hands_observed))
        return super().act(observation, legal_actions, rng)

    def observe_opponent_hand(
        self,
        counts: BehaviorCounts,
        *,
        learner_bot_id: int,
        opponent_bot_id: int,
        trajectory_hand_index: int,
    ) -> RecordedAdaptiveLearningEvent:
        self.timeline.append(("learn", self.model.hands_observed))
        return super().observe_opponent_hand(
            counts,
            learner_bot_id=learner_bot_id,
            opponent_bot_id=opponent_bot_id,
            trajectory_hand_index=trajectory_hand_index,
        )


def _normal_hand(actions: tuple[Action, ...] = (Action.FOLD,)) -> RecordedHand:
    return RecordedHand(
        config_id="A_BTN",
        master_seed=42,
        hand_index=0,
        dealer=0,
        seat0_bot_id=0,
        seat1_bot_id=1,
        actions=actions,
        termination=HandTermination.NORMAL,
        terminal_profit_bb=(-0.5, 0.5),
    )


def test_protocol_file_is_canonical_and_semantically_frozen() -> None:
    protocol = load_adaptive_lifecycle_protocol(PROTOCOL_PATH)
    assert protocol.protocol_version == ADAPTIVE_LIFECYCLE_PROTOCOL_VERSION
    assert protocol.payload_hash == ADAPTIVE_LIFECYCLE_PROTOCOL_HASH
    raw = PROTOCOL_PATH.read_bytes()
    assert b"\r" not in raw
    assert json.loads(raw)["payload_hash"] == ADAPTIVE_LIFECYCLE_PROTOCOL_HASH


def test_protocol_rejects_semantic_tampering_even_with_recomputed_hash(
    tmp_path: Path,
) -> None:
    document = json.loads(PROTOCOL_PATH.read_bytes())
    document["payload"]["runner"]["same_hand_adaptation"] = True
    canonical_payload = json.dumps(
        document["payload"],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    document["payload_hash"] = hashlib.sha256(canonical_payload).hexdigest()
    tampered_path = tmp_path / "phase2-adaptive-lifecycle-tampered.json"
    tampered_path.write_bytes(
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        + b"\n"
    )

    with pytest.raises(ValueError, match="hash differs from frozen v1"):
        load_adaptive_lifecycle_protocol(tampered_path)


def test_adaptive_agent_starts_from_frozen_prior_and_resets() -> None:
    reference = load_preflop_strength_reference(REFERENCE_PATH)
    agent = AdaptiveHandStrengthAgent(reference)
    initial = AdaptiveHandStrengthAgent.initial_model()

    assert agent.is_adaptive
    assert agent.version == ADAPTIVE_HAND_STRENGTH_AGENT_VERSION
    assert agent.model == initial

    counts = BehaviorCounts(vpip_opportunities=1, vpip_events=1)
    event = agent.observe_opponent_hand(
        counts,
        learner_bot_id=0,
        opponent_bot_id=1,
        trajectory_hand_index=0,
    )
    assert event.model_state_before_hash == initial.state_hash
    assert agent.model.hands_observed == 1

    agent.reset()
    assert agent.model == initial


def test_learning_event_hash_and_round_trip_are_strict() -> None:
    before = AdaptiveHandStrengthAgent.initial_model()
    counts = BehaviorCounts(vpip_opportunities=1, vpip_events=1)
    after = before.updated(counts, first_hand_index=0)
    event = RecordedAdaptiveLearningEvent.from_transition(
        learner_bot_id=0,
        opponent_bot_id=1,
        trajectory_hand_index=0,
        before=before,
        evidence=counts,
        after=after,
    )
    restored = RecordedAdaptiveLearningEvent.from_dict(event.to_dict())
    assert restored == event

    tampered = event.to_dict()
    tampered["model_state_after_hash"] = "0" * 64
    with pytest.raises(ValueError, match="hash"):
        RecordedAdaptiveLearningEvent.from_dict(tampered)


def test_recorded_hand_round_trip_preserves_learning_transition() -> None:
    before = AdaptiveHandStrengthAgent.initial_model()
    counts = BehaviorCounts(vpip_opportunities=1)
    after = before.updated(counts, first_hand_index=0)
    event = RecordedAdaptiveLearningEvent.from_transition(
        learner_bot_id=0,
        opponent_bot_id=1,
        trajectory_hand_index=0,
        before=before,
        evidence=counts,
        after=after,
    )
    hand = _normal_hand()
    hand = RecordedHand.from_dict({
        **hand.to_dict(),
        "adaptive_learning_events": [event.to_dict()],
    })
    assert hand.adaptive_learning_events == (event,)
    assert RecordedHand.from_dict(hand.to_dict()) == hand


def test_runner_resets_before_both_symmetric_adaptive_runs_and_updates_after_hands() -> None:
    adaptive = _LifecycleProbeAgent()
    static = _CallCheckAgent()
    runner = PairedRunner(
        agent_a=adaptive,
        agent_b=static,
        num_hands=2,
        seed=321,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
    )

    result = runner.run()

    assert result.statistical_unit is StatisticalUnit.TRAJECTORY_CLUSTER
    assert result.num_valid_pairs == 1
    assert adaptive.reset_calls == 2
    assert len(adaptive.transitions) == 4
    # Two hands in each mirror run, and each mirror starts from the prior.
    assert [event.trajectory_hand_index for event in adaptive.transitions] == [0, 1, 0, 1]
    assert adaptive.transitions[0].model_state_before_hash == adaptive.transitions[2].model_state_before_hash
    assert result.match_record is not None
    assert sum(len(h.adaptive_learning_events) for h in result.match_record.hands) == 4


def test_learning_replay_reclassifies_public_evidence_and_verifies_hash_chain() -> None:
    adaptive = _LifecycleProbeAgent()
    runner = PairedRunner(
        agent_a=adaptive,
        agent_b=_CallCheckAgent(),
        num_hands=2,
        seed=777,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
    )
    result = runner.run()
    assert result.match_record is not None

    replayed = replay_adaptive_learning_for_bot(result.match_record, bot_id=0)
    assert tuple(replayed) == ("A_BTN", "B_BTN")
    assert all(model.hands_observed == 2 for model in replayed.values())


@pytest.mark.parametrize(
    ("tampered_state", "expected_message"),
    (
        ("before", "before-state hash mismatch"),
        ("after", "after-state hash mismatch"),
    ),
)
def test_learning_replay_rejects_rehashed_state_hash_tampering(
    tampered_state: str,
    expected_message: str,
) -> None:
    adaptive = _LifecycleProbeAgent()
    result = PairedRunner(
        agent_a=adaptive,
        agent_b=_CallCheckAgent(),
        num_hands=1,
        seed=778,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
    ).run()
    assert result.match_record is not None

    first_hand = result.match_record.hands[0]
    original_event = first_hand.adaptive_learning_events[0]
    counts = BehaviorCounts(**dict(original_event.public_evidence_counts))
    before = AdaptiveHandStrengthAgent.initial_model()
    expected_after = before.updated(counts, first_hand_index=0)
    alternate_counts = replace(
        counts,
        vpip_events=1 - counts.vpip_events,
    )
    alternate_state = before.updated(alternate_counts, first_hand_index=0)
    tampered_event = RecordedAdaptiveLearningEvent.from_transition(
        learner_bot_id=original_event.learner_bot_id,
        opponent_bot_id=original_event.opponent_bot_id,
        trajectory_hand_index=original_event.trajectory_hand_index,
        before=alternate_state if tampered_state == "before" else before,
        evidence=counts,
        after=alternate_state if tampered_state == "after" else expected_after,
    )
    tampered_hand = replace(
        first_hand,
        adaptive_learning_events=(tampered_event,),
    )
    tampered_record = replace(
        result.match_record,
        hands=(tampered_hand, *result.match_record.hands[1:]),
    )

    with pytest.raises(ValueError, match=expected_message):
        replay_adaptive_learning_for_bot(tampered_record, bot_id=0)


def test_static_runner_does_not_add_learning_events() -> None:
    result = PairedRunner(
        agent_a=_CallCheckAgent(),
        agent_b=_CallCheckAgent(),
        num_hands=1,
        seed=15,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=False,
    ).run()
    assert result.match_record is not None
    for hand in result.match_record.hands:
        assert not hand.adaptive_learning_events
        assert "adaptive_learning_events" not in hand.to_dict()


def test_agent_observe_requires_one_vpip_opportunity_per_learnable_hand() -> None:
    agent = AdaptiveHandStrengthAgent(load_preflop_strength_reference(REFERENCE_PATH))
    with pytest.raises(ValueError, match="one VPIP opportunity"):
        agent.observe_opponent_hand(
            BehaviorCounts(),
            learner_bot_id=0,
            opponent_bot_id=1,
            trajectory_hand_index=0,
        )


def test_duplicate_or_out_of_order_learning_update_is_rejected() -> None:
    agent = AdaptiveHandStrengthAgent(load_preflop_strength_reference(REFERENCE_PATH))
    counts = BehaviorCounts(vpip_opportunities=1)
    agent.observe_opponent_hand(
        counts,
        learner_bot_id=0,
        opponent_bot_id=1,
        trajectory_hand_index=0,
    )
    with pytest.raises(ValueError, match="trajectory_hand_index"):
        agent.observe_opponent_hand(
            counts,
            learner_bot_id=0,
            opponent_bot_id=1,
            trajectory_hand_index=0,
        )
    with pytest.raises(ValueError, match="trajectory_hand_index"):
        agent.observe_opponent_hand(
            counts,
            learner_bot_id=0,
            opponent_bot_id=1,
            trajectory_hand_index=2,
        )


def test_timeout_altered_hand_cannot_be_used_as_adaptive_learning_evidence() -> None:
    before = AdaptiveHandStrengthAgent.initial_model()
    counts = BehaviorCounts(vpip_opportunities=1)
    after = before.updated(counts, first_hand_index=0)
    event = RecordedAdaptiveLearningEvent.from_transition(
        learner_bot_id=0,
        opponent_bot_id=1,
        trajectory_hand_index=0,
        before=before,
        evidence=counts,
        after=after,
    )
    payload = _normal_hand().to_dict()
    payload["termination"] = HandTermination.TIMEOUT_FORFEIT.value
    payload["adaptive_learning_events"] = [event.to_dict()]
    with pytest.raises(ValueError, match="timeout-altered"):
        RecordedHand.from_dict(payload)


def test_learning_event_rejects_non_opponent_identity() -> None:
    before = AdaptiveHandStrengthAgent.initial_model()
    counts = BehaviorCounts(vpip_opportunities=1)
    after = before.updated(counts, first_hand_index=0)
    with pytest.raises(ValueError, match="different"):
        RecordedAdaptiveLearningEvent.from_transition(
            learner_bot_id=0,
            opponent_bot_id=0,
            trajectory_hand_index=0,
            before=before,
            evidence=counts,
            after=after,
        )


def test_runner_invalidates_timeout_altered_adaptive_trajectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((0, 2_000_000))
    monkeypatch.setattr(
        runner_module.time,
        "perf_counter_ns",
        lambda: next(clock),
    )
    adaptive = _LifecycleProbeAgent()
    result = PairedRunner(
        agent_a=adaptive,
        agent_b=_CallCheckAgent(),
        num_hands=1,
        seed=999,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
        agent_a_budget=AgentExecutionBudget(
            time_budget_ms=1,
            timeout_policy=TimeoutPolicy.FOLD,
        ),
    ).run()

    assert result.num_valid_pairs == 0
    assert result.failed_runs
    assert result.failed_runs[0]["details"] == "adaptive_learning_evidence_invalid"
    assert adaptive.transitions == []
    assert result.match_record is not None
    assert result.match_record.hands[0].termination is HandTermination.TIMEOUT_FORFEIT
    assert result.match_record.hands[0].adaptive_learning_events == ()


def test_runner_does_not_expose_same_hand_adaptation() -> None:
    adaptive = _TimingProbeAgent()
    result = PairedRunner(
        agent_a=adaptive,
        agent_b=_CallCheckAgent(),
        num_hands=1,
        seed=2024,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
    ).run()

    assert result.num_valid_pairs == 1

    segments: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] | None = None
    for kind, observed_hands in adaptive.timeline:
        if kind == "reset":
            assert observed_hands == 0
            if current is not None:
                segments.append(current)
            current = []
            continue
        assert current is not None
        current.append((kind, observed_hands))
    assert current is not None
    segments.append(current)

    assert len(segments) == 2
    for segment in segments:
        assert segment[-1] == ("learn", 0)
        assert segment[:-1]
        assert all(item == ("act", 0) for item in segment[:-1])


def test_runner_preserves_stable_bot_identity_across_position_mirror() -> None:
    adaptive = _LifecycleProbeAgent()
    result = PairedRunner(
        agent_a=adaptive,
        agent_b=_CallCheckAgent(),
        num_hands=1,
        seed=2025,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
    ).run()

    assert result.match_record is not None
    hands = {hand.config_id: hand for hand in result.match_record.hands}
    assert (hands["A_BTN"].seat0_bot_id, hands["A_BTN"].seat1_bot_id) == (0, 1)
    assert (hands["B_BTN"].seat0_bot_id, hands["B_BTN"].seat1_bot_id) == (1, 0)

    for hand in hands.values():
        assert len(hand.adaptive_learning_events) == 1
        event = hand.adaptive_learning_events[0]
        assert event.learner_bot_id == 0
        assert event.opponent_bot_id == 1


def test_runner_updates_both_adaptive_agents_once_per_normal_hand() -> None:
    adaptive_a = _LifecycleProbeAgent()
    adaptive_b = _LifecycleProbeAgent()
    result = PairedRunner(
        agent_a=adaptive_a,
        agent_b=adaptive_b,
        num_hands=2,
        seed=2026,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
    ).run()

    assert result.num_valid_pairs == 1
    assert result.match_record is not None
    assert len(result.match_record.hands) == 4
    assert adaptive_a.reset_calls == 2
    assert adaptive_b.reset_calls == 2
    assert [event.trajectory_hand_index for event in adaptive_a.transitions] == [0, 1, 0, 1]
    assert [event.trajectory_hand_index for event in adaptive_b.transitions] == [0, 1, 0, 1]

    for hand in result.match_record.hands:
        assert hand.termination is HandTermination.NORMAL
        assert {
            (event.learner_bot_id, event.opponent_bot_id)
            for event in hand.adaptive_learning_events
        } == {(0, 1), (1, 0)}


def test_learning_replay_rejects_rehashed_public_evidence_tampering() -> None:
    adaptive = _LifecycleProbeAgent()
    result = PairedRunner(
        agent_a=adaptive,
        agent_b=_CallCheckAgent(),
        num_hands=1,
        seed=2027,
        game_config=BASELINE_GAME_CONFIG,
        adaptive=True,
    ).run()
    assert result.match_record is not None

    first_hand = result.match_record.hands[0]
    original_event = first_hand.adaptive_learning_events[0]
    original_counts = BehaviorCounts(**dict(original_event.public_evidence_counts))
    tampered_counts = replace(
        original_counts,
        vpip_events=1 - original_counts.vpip_events,
    )
    before = AdaptiveHandStrengthAgent.initial_model()
    after = before.updated(tampered_counts, first_hand_index=0)
    tampered_event = RecordedAdaptiveLearningEvent.from_transition(
        learner_bot_id=original_event.learner_bot_id,
        opponent_bot_id=original_event.opponent_bot_id,
        trajectory_hand_index=original_event.trajectory_hand_index,
        before=before,
        evidence=tampered_counts,
        after=after,
    )
    tampered_hand = replace(
        first_hand,
        adaptive_learning_events=(tampered_event,),
    )
    tampered_record = replace(
        result.match_record,
        hands=(tampered_hand, *result.match_record.hands[1:]),
    )

    with pytest.raises(ValueError, match="persisted public evidence differs"):
        replay_adaptive_learning_for_bot(tampered_record, bot_id=0)
