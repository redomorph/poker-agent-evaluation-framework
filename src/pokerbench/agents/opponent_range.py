"""Frozen posterior-to-preflop-range mapping for the first adaptive baseline."""
from __future__ import annotations
import hashlib, json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable
from pokerbench.agents.opponent_model import BayesianOpponentModel, OpponentModelMetric
from pokerbench.engine.config import Action
from pokerbench.engine.public_action import PublicActionEvent
from pokerbench.engine.state import Street

POSTERIOR_TO_RANGE_PROTOCOL_VERSION = "posterior-to-range-v1"
POSTERIOR_TO_RANGE_PROTOCOL_ID = "phase2-posterior-to-range-v1"
POSTERIOR_TO_RANGE_FILE_VERSION = "frozen-posterior-to-range-file-v1"
PREFLOP_RANGE_STRENGTH_REFERENCE_VERSION = "preflop-range-strength-reference-v1"
PREFLOP_RANGE_STRENGTH_REFERENCE_FILE_VERSION = "preflop-range-strength-reference-file-v1"
PREFLOP_RANGE_COMBO_ORDER_VERSION = "preflop-range-combo-order-v1"
PPM = 1_000_000
TOTAL_COMBOS = 1326
_RANKS = "23456789TJQKA"

class PreflopRangeSignal(StrEnum):
    UNINFORMATIVE = "uninformative"
    PASSIVE_ENTRY = "passive_entry"
    FIRST_RAISE = "first_raise"
    THREE_BET = "three_bet"

@dataclass(frozen=True)
class PreflopStrengthClass:
    class_id: str
    class_index: int
    representative_cards: tuple[int,int]
    equity_ppm: int
    sample_count: int

@dataclass(frozen=True)
class PreflopStrengthReference:
    reference_version: str
    payload_hash: str
    class_rows: tuple[PreflopStrengthClass,...]
    strength_order: tuple[str,...]
    source_equity_version: str
    source_equity_range_version: str
    source_samples_per_class: int
    reference_master_seed: int

@dataclass(frozen=True)
class EffectivePreflopWidths:
    raw_vpip_ppm: int
    raw_pfr_ppm: int
    raw_three_bet_ppm: int
    effective_vpip_ppm: int
    effective_pfr_ppm: int
    effective_three_bet_ppm: int

@dataclass(frozen=True)
class OpponentRangeSnapshot:
    signal: PreflopRangeSignal
    width_ppm: int
    selected_combo_count: int
    selected_combinations: tuple[tuple[int,int],...]
    model_state_hash: str
    reference_hash: str
    range_hash: str
    version: str = POSTERIOR_TO_RANGE_PROTOCOL_VERSION
    def legal_combinations(self, known_cards: Iterable[int]) -> tuple[tuple[int,int],...]:
        blocked = set(known_cards)
        for c in blocked:
            if type(c) is not int or not 0 <= c < 52:
                raise ValueError(f"invalid known card: {c!r}")
        legal = tuple(
            combo
            for combo in self.selected_combinations
            if combo[0] not in blocked and combo[1] not in blocked
        )
        if not legal:
            raise ValueError(
                "all selected opponent combinations are blocked; "
                "posterior-to-range-v1 forbids range backfill"
            )
        return legal

@dataclass(frozen=True)
class PosteriorToRangeProtocol:
    protocol_id: str
    protocol_version: str
    payload_hash: str
    reference_hash: str


def _canonical(payload: object) -> bytes:
    return json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode("utf-8")

def _canonical_file(payload: object) -> bytes:
    return _canonical(payload)+b"\n"

def _half_up_fraction_ppm(num: int, den: int) -> int:
    return (num*PPM + den//2)//den

def _posterior_ppm(model: BayesianOpponentModel, metric: OpponentModelMetric) -> int:
    p=model.posterior(metric)
    return _half_up_fraction_ppm(p.alpha,p.alpha+p.beta)

def effective_preflop_widths_ppm(model: BayesianOpponentModel) -> EffectivePreflopWidths:
    v=_posterior_ppm(model,OpponentModelMetric.VPIP)
    p=_posterior_ppm(model,OpponentModelMetric.PFR)
    t=_posterior_ppm(model,OpponentModelMetric.THREE_BET)
    ep=min(p,v); et=min(t,ep)
    return EffectivePreflopWidths(v,p,t,v,ep,et)

def _class_parts(class_id: str) -> tuple[int,int,str]:
    if len(class_id) not in (2,3): raise ValueError(f"invalid class id: {class_id}")
    try: a=_RANKS.index(class_id[0]); b=_RANKS.index(class_id[1])
    except ValueError as exc: raise ValueError(f"invalid class id: {class_id}") from exc
    if len(class_id)==2:
        if a!=b: raise ValueError(f"invalid pair class: {class_id}")
        return a,b,"p"
    if a<=b or class_id[2] not in "so": raise ValueError(f"invalid non-pair class: {class_id}")
    return a,b,class_id[2]

def canonical_class_combinations(class_id: str) -> tuple[tuple[int,int],...]:
    hi,lo,kind=_class_parts(class_id)
    combos=[]
    if kind=="p":
        cards=[s*13+hi for s in range(4)]
        for i in range(4):
            for j in range(i+1,4): combos.append(tuple(sorted((cards[i],cards[j]))))
    elif kind=="s":
        for s in range(4): combos.append(tuple(sorted((s*13+hi,s*13+lo))))
    else:
        for sh in range(4):
            for sl in range(4):
                if sh!=sl: combos.append(tuple(sorted((sh*13+hi,sl*13+lo))))
    return tuple(sorted(combos))

def expand_reference_combinations(reference: PreflopStrengthReference) -> tuple[tuple[int,int],...]:
    result=[]
    for cid in reference.strength_order: result.extend(canonical_class_combinations(cid))
    combos=tuple(result)
    if len(combos)!=TOTAL_COMBOS or len(set(combos))!=TOTAL_COMBOS: raise ValueError("reference does not expand to 1326 unique combinations")
    return combos

def load_preflop_strength_reference(path: Path) -> PreflopStrengthReference:
    raw=path.read_bytes(); doc=json.loads(raw)
    if raw!=_canonical_file(doc): raise ValueError("reference file is not canonical JSON/LF")
    if set(doc)!={"file_version","payload","payload_hash"}: raise ValueError("invalid reference document keys")
    if doc["file_version"]!=PREFLOP_RANGE_STRENGTH_REFERENCE_FILE_VERSION: raise ValueError("invalid reference file version")
    payload=doc["payload"]; h=hashlib.sha256(_canonical(payload)).hexdigest()
    if doc["payload_hash"]!=h: raise ValueError("reference payload hash mismatch")
    if payload["reference_version"]!=PREFLOP_RANGE_STRENGTH_REFERENCE_VERSION: raise ValueError("invalid reference version")
    if payload["source_equity_version"]!="uniform-random-equity-v1" or payload["source_equity_range_version"]!="uniform-random-range-v1" or payload["source_samples_per_class"]!=10000 or payload["reference_master_seed"]!=7200001: raise ValueError("reference generation contract mismatch")
    rows=tuple(PreflopStrengthClass(r["class_id"],r["class_index"],tuple(r["representative_cards"]),r["equity_ppm"],r["sample_count"]) for r in payload["rows"])
    if len(rows)!=169 or tuple(r.class_index for r in rows)!=tuple(range(169)): raise ValueError("invalid class enumeration")
    ids=tuple(r.class_id for r in rows); order=tuple(payload["strength_order"])
    if set(ids)!=set(order) or len(order)!=169: raise ValueError("invalid strength order")
    by={r.class_id:r for r in rows}
    expected=tuple(sorted(ids,key=lambda cid:(-by[cid].equity_ppm,cid)))
    if order!=expected: raise ValueError("strength order does not match scores/tie break")
    ref=PreflopStrengthReference(payload["reference_version"],h,rows,order,payload["source_equity_version"],payload["source_equity_range_version"],payload["source_samples_per_class"],payload["reference_master_seed"])
    expand_reference_combinations(ref)
    return ref

def classify_preflop_range_signal(history: tuple[PublicActionEvent,...], *, opponent_id: int) -> PreflopRangeSignal:
    if opponent_id not in (0,1): raise ValueError("opponent_id must be 0 or 1")
    signal=PreflopRangeSignal.UNINFORMATIVE; prior_increases=0
    for e in history:
        if e.street is not Street.PREFLOP: continue
        if e.actor==opponent_id:
            if e.wager_increase:
                if prior_increases==0: signal=PreflopRangeSignal.FIRST_RAISE
                elif prior_increases==1: signal=PreflopRangeSignal.THREE_BET
                # 4-bet+ does not invent a new v1 range; preserve existing signal.
            elif e.action in (Action.CALL,Action.ALL_IN) and signal is PreflopRangeSignal.UNINFORMATIVE:
                signal=PreflopRangeSignal.PASSIVE_ENTRY
        if e.wager_increase: prior_increases+=1
    return signal

def _width_for_signal(widths: EffectivePreflopWidths, signal: PreflopRangeSignal) -> int:
    if signal is PreflopRangeSignal.UNINFORMATIVE: return PPM
    if signal is PreflopRangeSignal.PASSIVE_ENTRY: return widths.effective_vpip_ppm
    if signal is PreflopRangeSignal.FIRST_RAISE: return widths.effective_pfr_ppm
    if signal is PreflopRangeSignal.THREE_BET: return widths.effective_three_bet_ppm
    raise ValueError("unsupported signal")

def construct_opponent_range(model: BayesianOpponentModel, signal: PreflopRangeSignal, reference: PreflopStrengthReference) -> OpponentRangeSnapshot:
    widths=effective_preflop_widths_ppm(model); width=_width_for_signal(widths,signal)
    n = max(1, (width * TOTAL_COMBOS + PPM - 1) // PPM)
    selected=expand_reference_combinations(reference)[:n]
    core={"version":POSTERIOR_TO_RANGE_PROTOCOL_VERSION,"signal":signal.value,"width_ppm":width,"selected_combo_count":n,"selected_combinations":[list(c) for c in selected],"model_state_hash":model.state_hash,"reference_hash":reference.payload_hash}
    rh=hashlib.sha256(_canonical(core)).hexdigest()
    return OpponentRangeSnapshot(signal,width,n,selected,model.state_hash,reference.payload_hash,rh)

def load_posterior_to_range_protocol(path: Path, *, reference_path: Path) -> PosteriorToRangeProtocol:
    raw=path.read_bytes(); doc=json.loads(raw)
    if raw!=_canonical_file(doc): raise ValueError("protocol file is not canonical JSON/LF")
    if set(doc)!={"file_version","payload","payload_hash"} or doc["file_version"]!=POSTERIOR_TO_RANGE_FILE_VERSION: raise ValueError("invalid protocol document")
    payload=doc["payload"]; h=hashlib.sha256(_canonical(payload)).hexdigest()
    if doc["payload_hash"]!=h: raise ValueError("protocol payload hash mismatch")
    expected={
      "protocol_id":POSTERIOR_TO_RANGE_PROTOCOL_ID,"protocol_version":POSTERIOR_TO_RANGE_PROTOCOL_VERSION,"status":"frozen","scope":"posterior-to-preflop-range-only",
      "dependencies":{"adaptive_learning_protocol":"adaptive-learning-protocol-v1","adaptive_learning_protocol_hash":"03f8ab5b2b35c86910d46df1d5187c306814e427b3f514eb85748a9a424adf5c","opponent_model":"bayesian-opponent-model-v1","public_action_trace":"public-action-trace-v1"},
      "posterior":{"statistic":"posterior-mean","ppm_scale":1000000,"ppm_rounding":"nearest-half-up-v1","range_consistency_projection":"three_bet<=pfr<=vpip-v1","raw_posteriors_unchanged":True},
      "preflop_signals":{"uninformative":"full-uniform-range","passive_entry":"effective-vpip","first_raise":"effective-pfr","three_bet":"effective-three-bet","four_bet_or_higher":"retain-last-v1-signal"},
      "reference":{"file":"configs/experiments/preflop-range-strength-reference-v1.json","payload_hash":"cb110768ad74a9747064ba25e88af635055b1491a58deae8f09112aae801fc1e","reference_version":"preflop-range-strength-reference-v1","source_equity_version":"uniform-random-equity-v1","source_samples_per_class":10000,"reference_master_seed":7200001},
      "quantile":{"population_combinations":1326,"selected_count":"max(1,ceil(width_ppm*1326/1000000))","class_order":"equity_ppm-desc-then-class_id-ascii","within_class_order":"canonical-card-tuple-ascending","combo_order_version":"preflop-range-combo-order-v1"},
      "blockers":{"application":"after-quantile-selection","backfill":False,"probability":"uniform-over-surviving-selected-combinations","empty_range_policy":"error-v1"},
      "postflop":{"carry_forward_last_preflop_range":True,"remove_new_board_blockers":True,"action_conditioned_narrowing":False},
      "unused_behavior_metrics_for_card_range":["cbet_freq","postflop_aggressive_share","fold_to_cbet"],
      "rng":{"range_construction_draws":0,"posterior_sampling":False},
      "non_goals":["posterior-to-action-mapping","postflop-action-conditioned-range","posterior-uncertainty-action-adjustment","h4-experiment-design","mcts-semantics"]}
    if payload!=expected: raise ValueError("protocol semantics differ from posterior-to-range-v1")
    ref=load_preflop_strength_reference(reference_path)
    if ref.payload_hash!=payload["reference"]["payload_hash"]: raise ValueError("reference hash does not match protocol")
    return PosteriorToRangeProtocol(payload["protocol_id"],payload["protocol_version"],h,ref.payload_hash)
