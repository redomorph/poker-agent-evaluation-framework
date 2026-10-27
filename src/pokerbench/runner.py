# pokerbench/runner.py
"""
Paired runner implementing CRN-based position compensation.

For static agents, each paired deal is an independent statistical unit.

For adaptive agents, the complete paired trajectory is the independent
statistical unit.

Every attempted hand is captured as a durable RecordedHand so the completed
PairedResult can be persisted and replayed independently of agent code.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import statistics
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

from pokerbench.agents.base import Agent
from pokerbench.agents.budget import (
    AgentExecutionBudget,
    TimeoutPolicy,
)
from pokerbench.engine.config import (
    Action,
    GameConfig,
    config_hash,
)
from pokerbench.engine.deal_schedule import (
    DealSchedule,
)
from pokerbench.engine.state import (
    GameState,
)
from pokerbench.replay.archive import (
    HandTermination,
    MatchRecord,
    RecordedHand,
    RecordedTimeoutEvent,
)
from pokerbench.persistence.base import (
    MatchStore,
)
from pokerbench.rng import AgentRNG


logger = logging.getLogger(
    __name__
)


class StatisticalUnit(Enum):
    PAIRED_DEAL = "paired_deal"
    INDEPENDENT_RUN = "independent_run"
    TRAJECTORY_CLUSTER = "trajectory_cluster"


class IllegalActionPolicy(Enum):
    EXCLUDE_RUN = "exclude_run"
    RAISE_EXCEPTION = "raise_exception"


@dataclass(frozen=True)
class TimeoutEvent:
    """One observed wall-clock agent-budget overrun."""

    hand_index: int
    config_id: str

    bot_id: int
    player: int

    elapsed_ns: int
    budget_ms: int

    timeout_policy: TimeoutPolicy
    timeout_policy_version: str

    resolution: str


@dataclass(frozen=True)
class PairedResult:
    paired_differences: List[float]

    statistical_unit: StatisticalUnit

    num_hands: int
    num_valid_pairs: int

    failed_runs: List[
        Dict[str, Any]
    ]

    experiment_metadata: Dict[
        str,
        Any,
    ]

    seed: int

    timeout_events: tuple[
        TimeoutEvent,
        ...,
    ] = field(
        default_factory=tuple
    )

    match_record: MatchRecord | None = field(
        default=None,
        repr=False,
    )

    mean_d: Optional[float] = field(
        default=None,
        repr=False,
    )

    std_d: Optional[float] = field(
        default=None,
        repr=False,
    )

    ci_low: Optional[float] = field(
        default=None,
        repr=False,
    )

    ci_high: Optional[float] = field(
        default=None,
        repr=False,
    )

    def __post_init__(
        self,
    ) -> None:
        if not self.paired_differences:
            return

        mean = (
            sum(
                self.paired_differences
            )
            / len(
                self.paired_differences
            )
        )

        object.__setattr__(
            self,
            "mean_d",
            mean,
        )

        if (
            len(
                self.paired_differences
            )
            <= 1
        ):
            return

        std = statistics.stdev(
            self.paired_differences
        )

        object.__setattr__(
            self,
            "std_d",
            std,
        )

        se = (
            std
            / math.sqrt(
                len(
                    self.paired_differences
                )
            )
        )

        ci_half_width = (
            1.96 * se
        )

        object.__setattr__(
            self,
            "ci_low",
            mean - ci_half_width,
        )

        object.__setattr__(
            self,
            "ci_high",
            mean + ci_half_width,
        )


class PairedRunner:
    def __init__(
        self,
        agent_a: Agent,
        agent_b: Agent,
        num_hands: int,
        seed: int,
        game_config: GameConfig,
        experiment_metadata: Optional[
            Dict[str, Any]
        ] = None,
        adaptive: Optional[
            bool
        ] = None,
        illegal_action_policy: IllegalActionPolicy = (
            IllegalActionPolicy.EXCLUDE_RUN
        ),
        agent_a_budget: Optional[
            AgentExecutionBudget
        ] = None,
        agent_b_budget: Optional[
            AgentExecutionBudget
        ] = None,
        match_id: str | None = None,
        benchmark_store: MatchStore | None = None,
    ) -> None:
        self.agent_a = agent_a
        self.agent_b = agent_b

        self.num_hands = num_hands
        self.seed = seed

        self.game_config = (
            game_config
        )

        self.experiment_metadata = (
            experiment_metadata
            or {}
        )

        self.adaptive = (
            adaptive
            if adaptive is not None
            else (
                agent_a.is_adaptive
                or agent_b.is_adaptive
            )
        )

        self.illegal_action_policy = (
            illegal_action_policy
        )

        self.agent_a_budget = (
            agent_a_budget
            if agent_a_budget
            is not None
            else AgentExecutionBudget()
        )

        self.agent_b_budget = (
            agent_b_budget
            if agent_b_budget
            is not None
            else AgentExecutionBudget()
        )

        if self.adaptive:
            self.statistical_unit = (
                StatisticalUnit.TRAJECTORY_CLUSTER
            )
        else:
            self.statistical_unit = (
                StatisticalUnit.PAIRED_DEAL
            )

        self.match_id = (
            match_id
            if match_id is not None
            else self._derive_match_id()
        )

        if not self.match_id:
            raise ValueError(
                "match_id must not be empty"
            )

        self.benchmark_store = (
            benchmark_store
        )

        self._timeout_events: list[
            TimeoutEvent
        ] = []

        self._recorded_hands: list[
            RecordedHand
        ] = []

    def run(
        self,
    ) -> PairedResult:
        """
        Execute the paired experiment.

        One deterministic deal schedule is generated for the complete run
        and reused by all paired conditions.
        """
        self._timeout_events = []
        self._recorded_hands = []

        deal_schedule = DealSchedule(
            master_seed=self.seed,
            num_hands=self.num_hands,
        )

        if self.adaptive:
            return self._run_adaptive(
                deal_schedule=(
                    deal_schedule
                ),
            )

        return self._run_static(
            deal_schedule=(
                deal_schedule
            ),
        )

    @staticmethod
    def _position_controlled_difference(
        profit_a_btn: float,
        profit_b_bb: float,
        profit_b_btn: float,
        profit_a_bb: float,
    ) -> float:
        d_btn = (
            profit_a_btn
            - profit_b_btn
        )

        d_bb = (
            profit_a_bb
            - profit_b_bb
        )

        return (
            d_btn
            + d_bb
        ) / 2.0

    def _run_static(
        self,
        deal_schedule: DealSchedule,
    ) -> PairedResult:
        paired_diffs: List[
            float
        ] = []

        failures: List[
            Dict[str, Any]
        ] = []

        for (
            hand_idx,
            deal,
        ) in enumerate(
            deal_schedule
        ):
            (
                hole_a,
                hole_b,
                board,
            ) = deal

            (
                profit_a_btn,
                profit_b_bb,
            ) = self._simulate_hand(
                agent_first=(
                    self.agent_a
                ),
                agent_second=(
                    self.agent_b
                ),
                hole_a=hole_a,
                hole_b=hole_b,
                board=board,
                dealer=0,
                hand_idx=hand_idx,
                config_id="A_BTN",
            )

            if (
                profit_a_btn is None
                or profit_b_bb is None
            ):
                failures.append(
                    {
                        "hand": hand_idx,
                        "config": "A_BTN",
                        "details": "failed",
                    }
                )

                continue

            (
                profit_b_btn,
                profit_a_bb,
            ) = self._simulate_hand(
                agent_first=(
                    self.agent_b
                ),
                agent_second=(
                    self.agent_a
                ),
                hole_a=hole_a,
                hole_b=hole_b,
                board=board,
                dealer=0,
                hand_idx=hand_idx,
                config_id="B_BTN",
            )

            if (
                profit_b_btn is None
                or profit_a_bb is None
            ):
                failures.append(
                    {
                        "hand": hand_idx,
                        "config": "B_BTN",
                        "details": "failed",
                    }
                )

                continue

            difference = (
                self._position_controlled_difference(
                    profit_a_btn=(
                        profit_a_btn
                    ),
                    profit_b_bb=(
                        profit_b_bb
                    ),
                    profit_b_btn=(
                        profit_b_btn
                    ),
                    profit_a_bb=(
                        profit_a_bb
                    ),
                )
            )

            paired_diffs.append(
                difference
            )

        return self._build_result(
            paired_differences=(
                paired_diffs
            ),
            num_valid_pairs=len(
                paired_diffs
            ),
            failed_runs=failures,
        )

    def _run_adaptive(
        self,
        deal_schedule: DealSchedule,
    ) -> PairedResult:
        # Every independent positional trajectory starts from the frozen
        # adaptive prior, including the first run of repeated runner use.
        self._reset_agents()

        (
            total_a_btn,
            total_b_bb,
            failures1,
        ) = self._run_full_match(
            agent_first=self.agent_a,
            agent_second=self.agent_b,
            deal_schedule=(
                deal_schedule
            ),
            dealer=0,
            config_id="A_BTN",
        )

        if (
            failures1
            or total_a_btn is None
            or total_b_bb is None
        ):
            return self._failed_result(
                failures1
            )

        self._reset_agents()

        (
            total_b_btn,
            total_a_bb,
            failures2,
        ) = self._run_full_match(
            agent_first=self.agent_b,
            agent_second=self.agent_a,
            deal_schedule=(
                deal_schedule
            ),
            dealer=0,
            config_id="B_BTN",
        )

        if (
            failures2
            or total_b_btn is None
            or total_a_bb is None
        ):
            return self._failed_result(
                failures2
            )

        difference = (
            self._position_controlled_difference(
                profit_a_btn=(
                    total_a_btn
                ),
                profit_b_bb=(
                    total_b_bb
                ),
                profit_b_btn=(
                    total_b_btn
                ),
                profit_a_bb=(
                    total_a_bb
                ),
            )
        )

        return self._build_result(
            paired_differences=[
                difference
            ],
            num_valid_pairs=1,
            failed_runs=[],
        )

    def _run_full_match(
        self,
        agent_first: Agent,
        agent_second: Agent,
        deal_schedule: DealSchedule,
        dealer: int,
        config_id: str,
    ) -> Tuple[
        Optional[float],
        Optional[float],
        List[
            Dict[str, Any]
        ],
    ]:
        total_first = 0.0
        total_second = 0.0

        failures: List[
            Dict[str, Any]
        ] = []

        for (
            hand_idx,
            deal,
        ) in enumerate(
            deal_schedule
        ):
            (
                hole_a,
                hole_b,
                board,
            ) = deal

            (
                profit_first,
                profit_second,
            ) = self._simulate_hand(
                agent_first=(
                    agent_first
                ),
                agent_second=(
                    agent_second
                ),
                hole_a=hole_a,
                hole_b=hole_b,
                board=board,
                dealer=dealer,
                hand_idx=hand_idx,
                config_id=(
                    config_id
                ),
            )

            if (
                profit_first is None
                or profit_second is None
            ):
                failures.append(
                    {
                        "hand": hand_idx,
                        "config": config_id,
                        "details": "failed",
                    }
                )

                break

            recorded_hand = self._recorded_hands[-1]
            if (
                self._has_adaptive_lifecycle_agent(
                    agent_first,
                    agent_second,
                )
                and (
                    recorded_hand.termination
                    is not HandTermination.NORMAL
                    or recorded_hand.timeout_events
                )
            ):
                failures.append(
                    {
                        "hand": hand_idx,
                        "config": config_id,
                        "details": (
                            "adaptive_learning_evidence_invalid"
                        ),
                    }
                )
                break

            total_first += (
                profit_first
            )

            total_second += (
                profit_second
            )

        if failures:
            return (
                None,
                None,
                failures,
            )

        return (
            total_first,
            total_second,
            failures,
        )

    def _failed_result(
        self,
        failures: List[
            Dict[str, Any]
        ],
    ) -> PairedResult:
        return self._build_result(
            paired_differences=[],
            num_valid_pairs=0,
            failed_runs=failures,
        )

    def _build_result(
        self,
        *,
        paired_differences: List[
            float
        ],
        num_valid_pairs: int,
        failed_runs: List[
            Dict[str, Any]
        ],
    ) -> PairedResult:
        match_record = (
            self._build_match_record()
        )

        self._persist_match_record(
            match_record
        )

        return PairedResult(
            paired_differences=(
                paired_differences
            ),
            statistical_unit=(
                self.statistical_unit
            ),
            num_hands=(
                self.num_hands
            ),
            num_valid_pairs=(
                num_valid_pairs
            ),
            failed_runs=(
                failed_runs
            ),
            experiment_metadata=(
                self._build_metadata(
                    match_record
                )
            ),
            seed=self.seed,
            timeout_events=tuple(
                self._timeout_events
            ),
            match_record=(
                match_record
            ),
        )

    def _persist_match_record(
        self,
        match_record: MatchRecord,
    ) -> None:
        """
        Persist the completed MatchRecord when a store was explicitly supplied.

        Persistence is deliberately outside hand simulation and statistical
        calculation. Store failures propagate to the caller: a benchmark must
        never appear successful when its requested durable record was not saved.
        """
        if self.benchmark_store is None:
            return

        self.benchmark_store.save_match(
            match_record
        )

    def _reset_agents(
        self,
    ) -> None:
        self.agent_a.reset()
        self.agent_b.reset()

    @staticmethod
    def _has_adaptive_lifecycle_agent(
        *agents: Agent,
    ) -> bool:
        return any(
            agent.is_adaptive
            and callable(
                getattr(
                    agent,
                    "observe_opponent_hand",
                    None,
                )
            )
            for agent in agents
        )

    def _with_adaptive_learning_events(
        self,
        *,
        hand: RecordedHand,
        agent_first: Agent,
        agent_second: Agent,
    ) -> RecordedHand:
        if not self.adaptive:
            return hand

        if (
            hand.termination
            is not HandTermination.NORMAL
            or hand.timeout_events
        ):
            return hand

        events = []
        for agent in (
            agent_first,
            agent_second,
        ):
            if not (
                agent.is_adaptive
                and callable(
                    getattr(
                        agent,
                        "observe_opponent_hand",
                        None,
                    )
                )
            ):
                continue

            from pokerbench.evaluation.opportunities import (
                measure_behavior_counts,
            )

            learner_bot_id = self._bot_id(
                agent
            )
            opponent_bot_id = (
                1 - learner_bot_id
            )
            counts = measure_behavior_counts(
                (hand,),
                self.game_config,
                bot_id=opponent_bot_id,
            )
            event = agent.observe_opponent_hand(
                counts,
                learner_bot_id=learner_bot_id,
                opponent_bot_id=opponent_bot_id,
                trajectory_hand_index=(
                    hand.hand_index
                ),
            )
            events.append(
                event
            )

        if not events:
            return hand

        return replace(
            hand,
            adaptive_learning_events=tuple(
                events
            ),
        )

    def _bot_id(
        self,
        agent: Agent,
    ) -> int:
        if agent is self.agent_a:
            return 0

        if agent is self.agent_b:
            return 1

        raise ValueError(
            "Agent is not one of the "
            "runner's registered agents"
        )

    def _budget_for(
        self,
        agent: Agent,
    ) -> AgentExecutionBudget:
        if agent is self.agent_a:
            return self.agent_a_budget

        if agent is self.agent_b:
            return self.agent_b_budget

        raise ValueError(
            "Agent is not one of the "
            "runner's registered agents"
        )

    @staticmethod
    def _forfeit_profit(
        state: GameState,
    ) -> tuple[
        float,
        float,
    ]:
        player = (
            state.current_player
        )

        opponent = (
            1 - player
        )

        contributions = (
            state.contributions
        )

        profit = [
            -contributions[0],
            -contributions[1],
        ]

        profit[opponent] += (
            state.pot
        )

        if (
            profit[0]
            + profit[1]
            != 0
        ):
            raise ValueError(
                "Timeout-forfeit profit "
                "must be zero-sum"
            )

        return (
            profit[0] / 100.0,
            profit[1] / 100.0,
        )

    def _resolve_timeout(
        self,
        *,
        state: GameState,
        legal_actions: list[
            Action
        ],
        timeout_policy: TimeoutPolicy,
    ) -> tuple[
        Optional[Action],
        Optional[
            tuple[
                float,
                float,
            ]
        ],
        str,
    ]:
        if (
            timeout_policy
            == TimeoutPolicy.CHECK_IF_LEGAL_ELSE_FOLD
            and Action.CHECK
            in legal_actions
        ):
            return (
                Action.CHECK,
                None,
                "check",
            )

        if timeout_policy in (
            TimeoutPolicy.FOLD,
            TimeoutPolicy.CHECK_IF_LEGAL_ELSE_FOLD,
        ):
            return (
                None,
                self._forfeit_profit(
                    state
                ),
                "forfeit",
            )

        raise ValueError(
            "Unsupported timeout policy"
        )

    def _record_timeout(
        self,
        *,
        agent: Agent,
        state: GameState,
        hand_idx: int,
        config_id: str,
        elapsed_ns: int,
        budget: AgentExecutionBudget,
        resolution: str,
    ) -> TimeoutEvent:
        if (
            budget.time_budget_ms
            is None
        ):
            raise ValueError(
                "Cannot record timeout without "
                "a declared time budget"
            )

        event = TimeoutEvent(
            hand_index=hand_idx,
            config_id=config_id,
            bot_id=self._bot_id(
                agent
            ),
            player=(
                state.current_player
            ),
            elapsed_ns=(
                elapsed_ns
            ),
            budget_ms=(
                budget.time_budget_ms
            ),
            timeout_policy=(
                budget.timeout_policy
            ),
            timeout_policy_version=(
                budget.timeout_policy_version
            ),
            resolution=(
                resolution
            ),
        )

        self._timeout_events.append(
            event
        )

        return event

    def _simulate_hand(
        self,
        agent_first: Agent,
        agent_second: Agent,
        hole_a: tuple[
            int,
            int,
        ],
        hole_b: tuple[
            int,
            int,
        ],
        board: tuple[
            int,
            int,
            int,
            int,
            int,
        ],
        dealer: int,
        hand_idx: int,
        config_id: str,
    ) -> Tuple[
        Optional[float],
        Optional[float],
    ]:
        state = GameState.new_hand(
            config=(
                self.game_config
            ),
            hole_cards=(
                hole_a,
                hole_b,
            ),
            board=board,
            dealer=dealer,
        )

        rng_first = AgentRNG(
            master_seed=self.seed,
            bot_id=self._bot_id(
                agent_first
            ),
            hand_id=hand_idx,
            purpose="decision",
        )

        rng_second = AgentRNG(
            master_seed=self.seed,
            bot_id=self._bot_id(
                agent_second
            ),
            hand_id=hand_idx,
            purpose="decision",
        )

        executed_actions: list[
            Action
        ] = []

        recorded_timeouts: list[
            RecordedTimeoutEvent
        ] = []

        while not state.is_terminal:
            player = (
                state.current_player
            )

            legal = (
                state.legal_actions()
            )

            if not legal:
                message = (
                    "Non-terminal state exposed "
                    "no legal actions"
                )

                self._append_recorded_hand(
                    agent_first=agent_first,
                    agent_second=agent_second,
                    dealer=dealer,
                    hand_idx=hand_idx,
                    config_id=config_id,
                    actions=executed_actions,
                    termination=(
                        HandTermination.ENGINE_ERROR
                    ),
                    terminal_profit_bb=None,
                    acting_player=player,
                    requested_action=None,
                    legal_actions=legal,
                    failure_message=message,
                    timeout_events=(
                        recorded_timeouts
                    ),
                )

                return (
                    None,
                    None,
                )

            observation = (
                state.observation(
                    player
                )
            )

            agent = (
                agent_first
                if player == 0
                else agent_second
            )

            rng = (
                rng_first
                if player == 0
                else rng_second
            )

            budget = (
                self._budget_for(
                    agent
                )
            )

            try:
                if (
                    budget.time_budget_ms
                    is None
                ):
                    action = agent.act(
                        observation,
                        legal,
                        rng,
                    )

                else:
                    start_ns = (
                        time.perf_counter_ns()
                    )

                    action = agent.act(
                        observation,
                        legal,
                        rng,
                    )

                    end_ns = (
                        time.perf_counter_ns()
                    )

                    elapsed_ns = (
                        end_ns
                        - start_ns
                    )

                    budget_ns = (
                        budget.time_budget_ms
                        * 1_000_000
                    )

                    if (
                        elapsed_ns
                        > budget_ns
                    ):
                        (
                            timeout_action,
                            timeout_profit,
                            resolution,
                        ) = self._resolve_timeout(
                            state=state,
                            legal_actions=legal,
                            timeout_policy=(
                                budget.timeout_policy
                            ),
                        )

                        timeout_event = (
                            self._record_timeout(
                                agent=agent,
                                state=state,
                                hand_idx=hand_idx,
                                config_id=config_id,
                                elapsed_ns=(
                                    elapsed_ns
                                ),
                                budget=budget,
                                resolution=(
                                    resolution
                                ),
                            )
                        )

                        recorded_timeouts.append(
                            RecordedTimeoutEvent(
                                action_index=len(
                                    executed_actions
                                ),
                                player=(
                                    timeout_event.player
                                ),
                                bot_id=(
                                    timeout_event.bot_id
                                ),
                                elapsed_ns=(
                                    timeout_event.elapsed_ns
                                ),
                                budget_ms=(
                                    timeout_event.budget_ms
                                ),
                                timeout_policy=(
                                    timeout_event.timeout_policy
                                ),
                                timeout_policy_version=(
                                    timeout_event.timeout_policy_version
                                ),
                                resolution=(
                                    timeout_event.resolution
                                ),
                            )
                        )

                        if (
                            timeout_profit
                            is not None
                        ):
                            self._append_recorded_hand(
                                agent_first=(
                                    agent_first
                                ),
                                agent_second=(
                                    agent_second
                                ),
                                dealer=dealer,
                                hand_idx=hand_idx,
                                config_id=config_id,
                                actions=(
                                    executed_actions
                                ),
                                termination=(
                                    HandTermination.TIMEOUT_FORFEIT
                                ),
                                terminal_profit_bb=(
                                    timeout_profit
                                ),
                                acting_player=player,
                                requested_action=(
                                    action
                                ),
                                legal_actions=(
                                    legal
                                ),
                                failure_message=None,
                                timeout_events=(
                                    recorded_timeouts
                                ),
                            )

                            return (
                                timeout_profit
                            )

                        if (
                            timeout_action
                            is None
                        ):
                            raise RuntimeError(
                                "Timeout resolution "
                                "produced no action "
                                "or terminal profit"
                            )

                        action = (
                            timeout_action
                        )

            except Exception as exc:
                message = (
                    f"{type(exc).__name__}: {exc}"
                )

                self._append_recorded_hand(
                    agent_first=agent_first,
                    agent_second=agent_second,
                    dealer=dealer,
                    hand_idx=hand_idx,
                    config_id=config_id,
                    actions=executed_actions,
                    termination=(
                        HandTermination.AGENT_ERROR
                    ),
                    terminal_profit_bb=None,
                    acting_player=player,
                    requested_action=None,
                    legal_actions=legal,
                    failure_message=message,
                    timeout_events=(
                        recorded_timeouts
                    ),
                )

                logger.error(
                    "Agent error in config %s, "
                    "hand %s: %s",
                    config_id,
                    hand_idx,
                    exc,
                )

                if (
                    self.illegal_action_policy
                    == IllegalActionPolicy.RAISE_EXCEPTION
                ):
                    raise

                return (
                    None,
                    None,
                )

            if action not in legal:
                message = (
                    f"Illegal action: {action}"
                )

                self._append_recorded_hand(
                    agent_first=agent_first,
                    agent_second=agent_second,
                    dealer=dealer,
                    hand_idx=hand_idx,
                    config_id=config_id,
                    actions=executed_actions,
                    termination=(
                        HandTermination.ILLEGAL_ACTION
                    ),
                    terminal_profit_bb=None,
                    acting_player=player,
                    requested_action=action,
                    legal_actions=legal,
                    failure_message=message,
                    timeout_events=(
                        recorded_timeouts
                    ),
                )

                logger.error(
                    "Illegal action %s from "
                    "agent %s in config %s, "
                    "hand %s",
                    action,
                    agent,
                    config_id,
                    hand_idx,
                )

                if (
                    self.illegal_action_policy
                    == IllegalActionPolicy.RAISE_EXCEPTION
                ):
                    raise ValueError(
                        message
                    )

                return (
                    None,
                    None,
                )

            try:
                new_state = (
                    state.apply_action(
                        action
                    )
                )

            except Exception as exc:
                message = (
                    f"{type(exc).__name__}: {exc}"
                )

                self._append_recorded_hand(
                    agent_first=agent_first,
                    agent_second=agent_second,
                    dealer=dealer,
                    hand_idx=hand_idx,
                    config_id=config_id,
                    actions=executed_actions,
                    termination=(
                        HandTermination.ENGINE_ERROR
                    ),
                    terminal_profit_bb=None,
                    acting_player=player,
                    requested_action=action,
                    legal_actions=legal,
                    failure_message=message,
                    timeout_events=(
                        recorded_timeouts
                    ),
                )

                logger.error(
                    "Apply-action error in "
                    "config %s, hand %s: %s",
                    config_id,
                    hand_idx,
                    exc,
                )

                if (
                    self.illegal_action_policy
                    == IllegalActionPolicy.RAISE_EXCEPTION
                ):
                    raise

                return (
                    None,
                    None,
                )

            executed_actions.append(
                action
            )

            state = (
                new_state
            )

        if (
            state.terminal_profit
            is None
        ):
            message = (
                "Terminal state has no terminal profit"
            )

            self._append_recorded_hand(
                agent_first=agent_first,
                agent_second=agent_second,
                dealer=dealer,
                hand_idx=hand_idx,
                config_id=config_id,
                actions=executed_actions,
                termination=(
                    HandTermination.ENGINE_ERROR
                ),
                terminal_profit_bb=None,
                acting_player=(
                    state.current_player
                ),
                requested_action=None,
                legal_actions=[],
                failure_message=message,
                timeout_events=(
                    recorded_timeouts
                ),
            )

            return (
                None,
                None,
            )

        terminal_profit = (
            state.terminal_profit[0]
            / 100.0,
            state.terminal_profit[1]
            / 100.0,
        )

        try:
            self._append_recorded_hand(
                agent_first=agent_first,
                agent_second=agent_second,
                dealer=dealer,
                hand_idx=hand_idx,
                config_id=config_id,
                actions=executed_actions,
                termination=(
                    HandTermination.NORMAL
                ),
                terminal_profit_bb=(
                    terminal_profit
                ),
                acting_player=None,
                requested_action=None,
                legal_actions=[],
                failure_message=None,
                timeout_events=(
                    recorded_timeouts
                ),
                apply_adaptive_learning=True,
            )
        except Exception as exc:
            message = (
                f"{type(exc).__name__}: {exc}"
            )
            self._append_recorded_hand(
                agent_first=agent_first,
                agent_second=agent_second,
                dealer=dealer,
                hand_idx=hand_idx,
                config_id=config_id,
                actions=executed_actions,
                termination=(
                    HandTermination.AGENT_ERROR
                ),
                terminal_profit_bb=None,
                acting_player=None,
                requested_action=None,
                legal_actions=[],
                failure_message=message,
                timeout_events=(
                    recorded_timeouts
                ),
                apply_adaptive_learning=False,
            )
            logger.error(
                "Adaptive learning error in config %s, hand %s: %s",
                config_id,
                hand_idx,
                exc,
            )
            if (
                self.illegal_action_policy
                == IllegalActionPolicy.RAISE_EXCEPTION
            ):
                raise
            return (
                None,
                None,
            )

        return terminal_profit

    def _append_recorded_hand(
        self,
        *,
        agent_first: Agent,
        agent_second: Agent,
        dealer: int,
        hand_idx: int,
        config_id: str,
        actions: list[
            Action
        ],
        termination: HandTermination,
        terminal_profit_bb: tuple[
            float,
            float,
        ] | None,
        acting_player: int | None,
        requested_action: Action | None,
        legal_actions: list[
            Action
        ],
        failure_message: str | None,
        timeout_events: list[
            RecordedTimeoutEvent
        ],
        apply_adaptive_learning: bool = False,
    ) -> None:
        recorded_hand = RecordedHand(
                config_id=config_id,
                master_seed=self.seed,
                hand_index=hand_idx,
                dealer=dealer,
                seat0_bot_id=self._bot_id(
                    agent_first
                ),
                seat1_bot_id=self._bot_id(
                    agent_second
                ),
                actions=tuple(
                    actions
                ),
                termination=(
                    termination
                ),
                terminal_profit_bb=(
                    terminal_profit_bb
                ),
                acting_player_at_termination=(
                    acting_player
                ),
                requested_action=(
                    requested_action
                ),
                legal_actions_at_termination=tuple(
                    legal_actions
                ),
                failure_message=(
                    failure_message
                ),
                timeout_events=tuple(
                    timeout_events
                ),
            )

        if apply_adaptive_learning:
            recorded_hand = (
                self._with_adaptive_learning_events(
                    hand=recorded_hand,
                    agent_first=agent_first,
                    agent_second=agent_second,
                )
            )

        self._recorded_hands.append(
            recorded_hand
        )

    def _build_match_record(
        self,
    ) -> MatchRecord:
        experiment_id = (
            self.experiment_metadata.get(
                "experiment_id"
            )
        )

        if not isinstance(
            experiment_id,
            str,
        ):
            experiment_id = None

        experiment_config_hash = (
            self.experiment_metadata.get(
                "experiment_config_hash"
            )
        )

        if not isinstance(
            experiment_config_hash,
            str,
        ):
            experiment_config_hash = None

        return MatchRecord(
            match_id=(
                self.match_id
            ),
            experiment_id=(
                experiment_id
            ),
            experiment_config_hash=(
                experiment_config_hash
            ),
            master_seed=(
                self.seed
            ),
            game_config=(
                self.game_config
            ),
            hands=tuple(
                self._recorded_hands
            ),
        )

    def _derive_match_id(
        self,
    ) -> str:
        """
        Derive a reproducible default match ID.

        Identical scientific inputs intentionally produce the same default
        ID. Callers may provide an explicit match_id when they need a
        separate operational identifier for an otherwise identical rerun.
        """
        material = {
            "seed": self.seed,
            "num_hands": (
                self.num_hands
            ),
            "game_config_hash": (
                config_hash(
                    self.game_config
                )
            ),
            "adaptive": (
                self.adaptive
            ),
            "agent_a_class": (
                f"{type(self.agent_a).__module__}."
                f"{type(self.agent_a).__qualname__}"
            ),
            "agent_b_class": (
                f"{type(self.agent_b).__module__}."
                f"{type(self.agent_b).__qualname__}"
            ),
            "agent_a_name": (
                getattr(
                    self.agent_a,
                    "name",
                    None,
                )
            ),
            "agent_b_name": (
                getattr(
                    self.agent_b,
                    "name",
                    None,
                )
            ),
            "experiment_id": (
                self.experiment_metadata.get(
                    "experiment_id"
                )
            ),
        }

        canonical = json.dumps(
            material,
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
            ensure_ascii=False,
            default=str,
        )

        digest = hashlib.sha256(
            canonical.encode(
                "utf-8"
            )
        ).hexdigest()

        return (
            f"match-{digest[:24]}"
        )

    def _build_metadata(
        self,
        match_record: MatchRecord,
    ) -> Dict[
        str,
        Any,
    ]:
        metadata: Dict[
            str,
            Any,
        ] = {
            "seed": (
                self.seed
            ),
            "num_hands": (
                self.num_hands
            ),
            "adaptive": (
                self.adaptive
            ),
            "statistical_unit": (
                self.statistical_unit.value
            ),
            "illegal_action_policy": (
                self.illegal_action_policy.value
            ),
            "agent_a_budget": (
                self.agent_a_budget.to_metadata()
            ),
            "agent_b_budget": (
                self.agent_b_budget.to_metadata()
            ),
            "timeout_event_count": (
                len(
                    self._timeout_events
                )
            ),
            "game_config_hash": (
                config_hash(
                    self.game_config
                )
            ),
            "match_id": (
                match_record.match_id
            ),
            "match_record_hash": (
                match_record.record_hash
            ),
            "replay_hand_count": (
                len(
                    match_record.hands
                )
            ),
            **self.experiment_metadata,
        }

        return metadata