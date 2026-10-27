from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from pokerbench.engine.betting_rules import BettingRules
from pokerbench.engine.config import Action, GameConfig
from pokerbench.engine.constants import CHIP_UNIT
from pokerbench.engine.hand_evaluator import HandEvaluator, compare_hands
from pokerbench.engine.public_action import PublicActionEvent

if TYPE_CHECKING:
    from pokerbench.agents.observation import AgentObservation


class Street(Enum):
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()
    SHOWDOWN = auto()


@dataclass(frozen=True)
class GameState:
    """
    Immutable true game state.

    The state contains complete game information, including both players'
    hole cards. Agent-visible information is exposed only through
    observation().
    """

    config: GameConfig
    hole_cards: tuple[tuple[int, int], tuple[int, int]]
    board: tuple[int, int, int, int, int]
    stacks: tuple[int, int]
    pot: int
    current_bets: tuple[int, int]
    current_player: int
    street: Street
    dealer: int
    action_history: tuple[Action, ...]
    street_actions: tuple[Action, ...]
    pot_at_street_start: int
    last_raise: int
    last_aggressor: int | None
    is_terminal: bool
    terminal_profit: tuple[int, int] | None = None
    public_action_history: tuple[PublicActionEvent, ...] = ()

    def _rules(self) -> BettingRules:
        return BettingRules.from_config(self.config)

    @property
    def visible_board(self) -> tuple[int, ...]:
        if self.street == Street.PREFLOP:
            return ()

        if self.street == Street.FLOP:
            return self.board[:3]

        if self.street == Street.TURN:
            return self.board[:4]

        return self.board

    @property
    def starting_stack(self) -> int:
        return int(self.config.starting_stack_bb * CHIP_UNIT)

    @property
    def contributions(self) -> tuple[int, int]:
        """
        Return each player's total contribution to the current hand.

        The baseline has no rake and no side pots, so total contributions
        equal starting stacks minus current remaining stacks.
        """
        return (
            self.starting_stack - self.stacks[0],
            self.starting_stack - self.stacks[1],
        )

    def observation(self, player_id: int) -> AgentObservation:
        """Return only the information observable by the specified player."""
        from pokerbench.agents.observation import AgentObservation

        if player_id not in (0, 1):
            raise ValueError("player_id must be 0 or 1")

        return AgentObservation(
            player_id=player_id,
            hole_cards=self.hole_cards[player_id],
            board=self.visible_board,
            stacks=self.stacks,
            current_bets=self.current_bets,
            pot=self.pot,
            current_player=self.current_player,
            street=self.street,
            action_history=self.action_history,
            legal_actions=tuple(self.legal_actions()),
            dealer=self.dealer,
            public_action_history=self.public_action_history,
        )

    def legal_actions(self) -> list[Action]:
        if self.is_terminal:
            return []

        player = self.current_player
        opponent = 1 - player
        stack = self.stacks[player]
        current_bet = self.current_bets[player]
        opponent_bet = self.current_bets[opponent]
        diff = opponent_bet - current_bet

        if stack == 0:
            return []

        actions: list[Action] = []

        if diff > 0:
            actions.append(Action.FOLD)

            if stack >= diff:
                actions.append(Action.CALL)
        else:
            actions.append(Action.CHECK)

        if self._is_all_in_legal():
            actions.append(Action.ALL_IN)

        for action in (
            Action.BET_33,
            Action.BET_66,
            Action.BET_100,
            Action.BET_200,
        ):
            if self._is_bet_legal(action):
                actions.append(action)

        return actions

    def _is_all_in_legal(self) -> bool:
        player = self.current_player
        opponent = 1 - player

        stack = self.stacks[player]
        current_bet = self.current_bets[player]
        opponent_bet = self.current_bets[opponent]

        if stack <= 0:
            return False

        diff = opponent_bet - current_bet

        if diff > 0:
            if stack < diff:
                return True

            target = current_bet + stack
            min_raise = opponent_bet + self._rules().min_raise_increment(
                self.last_raise
            )

            return target >= min_raise

        return stack >= self._rules().min_bet()

    def _is_bet_legal(self, action: Action) -> bool:
        player = self.current_player
        opponent = 1 - player

        stack = self.stacks[player]
        current_bet = self.current_bets[player]
        opponent_bet = self.current_bets[opponent]
        diff = opponent_bet - current_bet

        rules = self._rules()

        if diff > 0:
            target = rules.raise_to_amount(
                action=action,
                pot_at_street_start=self.pot_at_street_start,
                opponent_bet=opponent_bet,
                current_bet=current_bet,
                stack=stack,
            )

            if target is None:
                return False

            if target <= opponent_bet:
                return False

            min_raise = opponent_bet + rules.min_raise_increment(
                self.last_raise
            )

            if target < min_raise:
                return False

            if target > current_bet + stack:
                return False

            return True

        amount = rules.bet_amount(
            action=action,
            pot_at_street_start=self.pot_at_street_start,
            stack=stack,
        )

        if amount is None or amount <= 0:
            return False

        if amount < rules.min_bet():
            return False

        return amount <= stack

    def _bet_amount(self, action: Action) -> int | None:
        if action not in (
            Action.BET_33,
            Action.BET_66,
            Action.BET_100,
            Action.BET_200,
        ):
            return None

        player = self.current_player
        opponent = 1 - player

        current_bet = self.current_bets[player]
        opponent_bet = self.current_bets[opponent]
        stack = self.stacks[player]

        rules = self._rules()

        if opponent_bet > current_bet:
            target = rules.raise_to_amount(
                action=action,
                pot_at_street_start=self.pot_at_street_start,
                opponent_bet=opponent_bet,
                current_bet=current_bet,
                stack=stack,
            )

            if target is None:
                return None

            return target - current_bet

        return rules.bet_amount(
            action=action,
            pot_at_street_start=self.pot_at_street_start,
            stack=stack,
        )

    def apply_action(self, action: Action) -> GameState:
        if self.is_terminal:
            raise ValueError("Cannot apply action to terminal state")

        if action not in self.legal_actions():
            raise ValueError(
                f"Action {action} is not legal in current state"
            )

        if action == Action.FOLD:
            return self._apply_fold()

        if action == Action.CHECK:
            return self._apply_check()

        if action == Action.CALL:
            return self._apply_call()

        if action == Action.ALL_IN:
            return self._apply_all_in()

        if action in (
            Action.BET_33,
            Action.BET_66,
            Action.BET_100,
            Action.BET_200,
        ):
            amount = self._bet_amount(action)

            if amount is None:
                raise ValueError(
                    f"Could not compute amount for {action}"
                )

            return self._apply_bet_raise(action, amount)

        raise ValueError(f"Unknown action: {action}")

    def _apply_fold(self) -> GameState:
        player = self.current_player
        opponent = 1 - player

        contributions = self.contributions

        profit = [
            -contributions[0],
            -contributions[1],
        ]

        profit[opponent] += self.pot

        return self._terminal_state(
            (profit[0], profit[1]),
            public_action_event=(
                PublicActionEvent(
                    actor=self.current_player,
                    street=self.street,
                    action=Action.FOLD,
                    wager_increase=False,
                )
            ),
        )

    def _apply_check(self) -> GameState:
        new_state = GameState(
            config=self.config,
            hole_cards=self.hole_cards,
            board=self.board,
            stacks=self.stacks,
            pot=self.pot,
            current_bets=self.current_bets,
            current_player=self.current_player,
            street=self.street,
            dealer=self.dealer,
            action_history=self.action_history + (Action.CHECK,),
            street_actions=self.street_actions + (Action.CHECK,),
            pot_at_street_start=self.pot_at_street_start,
            last_raise=self.last_raise,
            last_aggressor=self.last_aggressor,
            is_terminal=False,
            terminal_profit=None,
            public_action_history=(
                self.public_action_history
                + (
                    PublicActionEvent(
                        actor=self.current_player,
                        street=self.street,
                        action=Action.CHECK,
                        wager_increase=False,
                    ),
                )
            ),
        )

        new_state = new_state._switch_player()

        if new_state._is_round_complete():
            return new_state._advance_street()

        return new_state

    def _apply_call(self) -> GameState:
        player = self.current_player
        opponent = 1 - player

        diff = (
            self.current_bets[opponent]
            - self.current_bets[player]
        )

        if diff <= 0:
            raise ValueError("No bet to call")

        new_stacks = list(self.stacks)
        new_stacks[player] -= diff

        new_bets = list(self.current_bets)
        new_bets[player] += diff

        new_state = GameState(
            config=self.config,
            hole_cards=self.hole_cards,
            board=self.board,
            stacks=(new_stacks[0], new_stacks[1]),
            pot=self.pot + diff,
            current_bets=(new_bets[0], new_bets[1]),
            current_player=opponent,
            street=self.street,
            dealer=self.dealer,
            action_history=self.action_history + (Action.CALL,),
            street_actions=self.street_actions + (Action.CALL,),
            pot_at_street_start=self.pot_at_street_start,
            last_raise=self.last_raise,
            last_aggressor=self.last_aggressor,
            is_terminal=False,
            terminal_profit=None,
            public_action_history=(
                self.public_action_history
                + (
                    PublicActionEvent(
                        actor=self.current_player,
                        street=self.street,
                        action=Action.CALL,
                        wager_increase=False,
                    ),
                )
            ),
        )

        if new_state._is_round_complete():
            return new_state._advance_street()

        return new_state

    def _apply_all_in(self) -> GameState:
        player = self.current_player
        opponent = 1 - player
        stack = self.stacks[player]

        if stack <= 0:
            raise ValueError("Cannot go all-in with zero stack")

        new_stacks = list(self.stacks)
        new_stacks[player] = 0

        new_bets = list(self.current_bets)
        new_bets[player] += stack

        new_current_bet = new_bets[player]
        opponent_bet = self.current_bets[opponent]

        last_raise = self.last_raise
        last_aggressor = self.last_aggressor

        if new_current_bet > opponent_bet:
            last_raise = new_current_bet - opponent_bet
            last_aggressor = player

        new_state = GameState(
            config=self.config,
            hole_cards=self.hole_cards,
            board=self.board,
            stacks=(new_stacks[0], new_stacks[1]),
            pot=self.pot + stack,
            current_bets=(new_bets[0], new_bets[1]),
            current_player=opponent,
            street=self.street,
            dealer=self.dealer,
            action_history=self.action_history + (Action.ALL_IN,),
            street_actions=self.street_actions + (Action.ALL_IN,),
            pot_at_street_start=self.pot_at_street_start,
            last_raise=last_raise,
            last_aggressor=last_aggressor,
            is_terminal=False,
            terminal_profit=None,
            public_action_history=(
                self.public_action_history
                + (
                    PublicActionEvent(
                        actor=self.current_player,
                        street=self.street,
                        action=Action.ALL_IN,
                        wager_increase=(
                            new_current_bet
                            > opponent_bet
                        ),
                    ),
                )
            ),
        )

        return new_state

    def _apply_bet_raise(
        self,
        action: Action,
        amount: int,
    ) -> GameState:
        player = self.current_player
        opponent = 1 - player

        if amount <= 0 or amount > self.stacks[player]:
            raise ValueError(f"Invalid bet amount: {amount}")

        new_stacks = list(self.stacks)
        new_stacks[player] -= amount

        new_bets = list(self.current_bets)
        new_bets[player] += amount

        opponent_bet = self.current_bets[opponent]
        new_current_bet = new_bets[player]

        if new_current_bet <= opponent_bet:
            raise ValueError("Bet or raise must increase the current wager")

        last_raise = new_current_bet - opponent_bet
        last_aggressor = player

        new_state = GameState(
            config=self.config,
            hole_cards=self.hole_cards,
            board=self.board,
            stacks=(new_stacks[0], new_stacks[1]),
            pot=self.pot + amount,
            current_bets=(new_bets[0], new_bets[1]),
            current_player=opponent,
            street=self.street,
            dealer=self.dealer,
            action_history=self.action_history + (action,),
            street_actions=self.street_actions + (action,),
            pot_at_street_start=self.pot_at_street_start,
            last_raise=last_raise,
            last_aggressor=last_aggressor,
            is_terminal=False,
            terminal_profit=None,
            public_action_history=(
                self.public_action_history
                + (
                    PublicActionEvent(
                        actor=self.current_player,
                        street=self.street,
                        action=action,
                        wager_increase=True,
                    ),
                )
            ),
        )

        if new_state._is_round_complete():
            return new_state._advance_street()

        return new_state

    def _is_round_complete(self) -> bool:
        if self.is_terminal:
            return True

        # An all-in does NOT by itself complete the betting round.
        # The opponent must still be able to call or fold.
        #
        # The round becomes complete once the wagers are equal and
        # the final action is CALL/CHECK. This also correctly handles
        # the all-in + call case.
        if self.current_bets[0] != self.current_bets[1]:
            return False

        if len(self.street_actions) < 2:
            return False

        return self.street_actions[-1] in (
            Action.CHECK,
            Action.CALL,
        )

    def _postflop_first_player(self) -> int:
        """
        Heads-Up postflop action starts with the Big Blind.

        The dealer is simultaneously the Small Blind in heads-up.
        Therefore the first postflop actor is always the other player.
        """
        return 1 - self.dealer

    def _advance_street(self) -> GameState:
        if self.is_terminal:
            return self

        # If either player has no chips left, all remaining betting is
        # complete and the hand must go directly to showdown.
        #
        # This is intentionally checked before normal street advancement.
        # Otherwise an all-in + call on the preflop would incorrectly
        # advance to the flop and allow further actions.
        if self.stacks[0] == 0 or self.stacks[1] == 0:
            return self._terminal_showdown()

        if self.street == Street.PREFLOP:
            new_street = Street.FLOP
        elif self.street == Street.FLOP:
            new_street = Street.TURN
        elif self.street == Street.TURN:
            new_street = Street.RIVER
        else:
            new_street = Street.SHOWDOWN

        if new_street == Street.SHOWDOWN:
            return self._terminal_showdown()

        return GameState(
            config=self.config,
            hole_cards=self.hole_cards,
            board=self.board,
            stacks=self.stacks,
            pot=self.pot,
            current_bets=(0, 0),
            current_player=self._postflop_first_player(),
            street=new_street,
            dealer=self.dealer,
            action_history=self.action_history,
            street_actions=(),
            pot_at_street_start=self.pot,
            last_raise=0,
            last_aggressor=None,
            is_terminal=False,
            terminal_profit=None,
            public_action_history=self.public_action_history,
        )

    def _terminal_showdown(self) -> GameState:
        hand0 = self.hole_cards[0] + self.board
        hand1 = self.hole_cards[1] + self.board

        best0 = HandEvaluator.best_hand(hand0)
        best1 = HandEvaluator.best_hand(hand1)

        cmp = compare_hands(best0, best1)

        contributions = self.contributions

        if cmp > 0:
            profit = (
                self.pot - contributions[0],
                -contributions[1],
            )
        elif cmp < 0:
            profit = (
                -contributions[0],
                self.pot - contributions[1],
            )
        else:
            half = self.pot // 2
            profit = (
                half - contributions[0],
                self.pot - half - contributions[1],
            )

        return self._terminal_state(profit)

    def _terminal_state(
        self,
        profit: tuple[int, int],
        public_action_event: PublicActionEvent | None = None,
    ) -> GameState:
        if profit[0] + profit[1] != 0:
            raise ValueError(
                "Terminal profit must be zero-sum"
            )

        return GameState(
            config=self.config,
            hole_cards=self.hole_cards,
            board=self.board,
            stacks=self.stacks,
            pot=self.pot,
            current_bets=self.current_bets,
            current_player=self.current_player,
            street=self.street,
            dealer=self.dealer,
            action_history=self.action_history,
            street_actions=self.street_actions,
            pot_at_street_start=self.pot_at_street_start,
            last_raise=self.last_raise,
            last_aggressor=self.last_aggressor,
            is_terminal=True,
            terminal_profit=profit,
            public_action_history=(
                self.public_action_history
                if public_action_event is None
                else (
                    self.public_action_history
                    + (public_action_event,)
                )
            ),
        )

    def _switch_player(self) -> GameState:
        return GameState(
            config=self.config,
            hole_cards=self.hole_cards,
            board=self.board,
            stacks=self.stacks,
            pot=self.pot,
            current_bets=self.current_bets,
            current_player=1 - self.current_player,
            street=self.street,
            dealer=self.dealer,
            action_history=self.action_history,
            street_actions=self.street_actions,
            pot_at_street_start=self.pot_at_street_start,
            last_raise=self.last_raise,
            last_aggressor=self.last_aggressor,
            is_terminal=self.is_terminal,
            terminal_profit=self.terminal_profit,
            public_action_history=self.public_action_history,
        )

    def next_hand(
        self,
        hole_cards: tuple[tuple[int, int], tuple[int, int]],
        board: tuple[int, int, int, int, int],
    ) -> GameState:
        """
        Start the next hand with a reset stack and rotated positions.

        The baseline resets both players to the configured starting stack
        and alternates the dealer/button between the two players.
        """
        if not self.is_terminal:
            raise ValueError(
                "Cannot start the next hand before the current hand ends"
            )

        return self.new_hand(
            config=self.config,
            hole_cards=hole_cards,
            board=board,
            dealer=1 - self.dealer,
        )

    @classmethod
    def new_hand(
        cls,
        config: GameConfig,
        hole_cards: tuple[tuple[int, int], tuple[int, int]],
        board: tuple[int, int, int, int, int],
        dealer: int = 0,
    ) -> GameState:
        if dealer not in (0, 1):
            raise ValueError("dealer must be 0 or 1")

        if len(hole_cards) != 2:
            raise ValueError("Exactly two players are required")

        if len(board) != 5:
            raise ValueError("Board must contain exactly five cards")

        sb = int(config.small_blind_bb * CHIP_UNIT)
        bb = int(config.big_blind_bb * CHIP_UNIT)
        starting_stack = int(
            config.starting_stack_bb * CHIP_UNIT
        )

        if dealer == 0:
            stacks = (
                starting_stack - sb,
                starting_stack - bb,
            )
            current_bets = (sb, bb)
        else:
            stacks = (
                starting_stack - bb,
                starting_stack - sb,
            )
            current_bets = (bb, sb)

        pot = sb + bb

        return cls(
            config=config,
            hole_cards=hole_cards,
            board=board,
            stacks=stacks,
            pot=pot,
            current_bets=current_bets,
            current_player=dealer,
            street=Street.PREFLOP,
            dealer=dealer,
            action_history=(),
            street_actions=(),
            pot_at_street_start=pot,
            last_raise=0,
            last_aggressor=None,
            is_terminal=False,
            terminal_profit=None,
        )