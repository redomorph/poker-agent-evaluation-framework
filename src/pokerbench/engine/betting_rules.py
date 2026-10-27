from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import Optional

from pokerbench.engine.config import Action, GameConfig
from pokerbench.engine.constants import CHIP_UNIT


class BettingRulesVersion(StrEnum):
    BASELINE_V1 = "baseline-v1"


@dataclass(frozen=True)
class BettingRules:
    """
    Versioned deterministic betting semantics for the baseline abstraction.

    BET_k represents k times the pot at the beginning of the street when
    no wager exists yet. When facing an opponent wager, the same action
    represents a deterministic raise-to amount.

    All amounts are integer chip units.
    """

    config: GameConfig
    version: BettingRulesVersion = BettingRulesVersion.BASELINE_V1

    @classmethod
    def from_config(cls, config: GameConfig) -> "BettingRules":
        if config.abstraction_version != BettingRulesVersion.BASELINE_V1:
            raise ValueError(
                f"Unsupported betting rules version: "
                f"{config.abstraction_version}"
            )

        return cls(
            config=config,
            version=BettingRulesVersion.BASELINE_V1,
        )

    @property
    def chip_big_blind(self) -> int:
        return int(self.config.big_blind_bb * CHIP_UNIT)

    def multiplier_for(self, action: Action) -> Optional[Decimal]:
        multipliers = {
            Action.BET_33: Decimal("0.33"),
            Action.BET_66: Decimal("0.66"),
            Action.BET_100: Decimal("1"),
            Action.BET_200: Decimal("2"),
        }

        return multipliers.get(action)

    def bet_amount(
        self,
        action: Action,
        pot_at_street_start: int,
        stack: int,
    ) -> Optional[int]:
        """
        Return the deterministic bet amount for an unopened street.

        The reference pot is the pot at the beginning of the street.
        The resulting amount is capped at the player's remaining stack.
        """
        multiplier = self.multiplier_for(action)

        if multiplier is None:
            return None

        if pot_at_street_start < 0:
            raise ValueError("pot_at_street_start must be non-negative")

        if stack < 0:
            raise ValueError("stack must be non-negative")

        amount = int(
            multiplier * Decimal(pot_at_street_start)
        )

        return min(max(0, amount), stack)

    def raise_to_amount(
        self,
        action: Action,
        pot_at_street_start: int,
        opponent_bet: int,
        current_bet: int,
        stack: int,
    ) -> Optional[int]:
        """
        Return the deterministic raise-to amount.

        The abstract multiplier is applied to the street-start pot and
        added to the opponent's current wager. The resulting target is
        capped at the maximum amount the acting player can have committed
        on the street.
        """
        multiplier = self.multiplier_for(action)

        if multiplier is None:
            return None

        if pot_at_street_start < 0:
            raise ValueError("pot_at_street_start must be non-negative")

        if opponent_bet < 0:
            raise ValueError("opponent_bet must be non-negative")

        if current_bet < 0:
            raise ValueError("current_bet must be non-negative")

        if stack < 0:
            raise ValueError("stack must be non-negative")

        target = opponent_bet + int(
            multiplier * Decimal(pot_at_street_start)
        )

        maximum_target = current_bet + stack

        return min(target, maximum_target)

    def min_bet(self) -> int:
        return self.chip_big_blind

    def min_raise_increment(self, last_raise: int) -> int:
        if last_raise > 0:
            return last_raise

        return self.min_bet()