from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
import pandas as pd


# Acc-based drift detector using Page-Hinkley test
@dataclass
class PageHinkleyDriftDetector:
    burn_in_windows: int = 6  # number of initial windows to establish reference CER (average over burn-in)
    lambda_ph: float = 0.05  # sensitivity threshold for PH statistic to signal drift (tunable)

    # state
    _cer_history: List[float] = field(default_factory=list)
    _reference_cer: Optional[float] = None
    _sum_m: float = 0.0
    _min_sum_m: float = 0.0

    def update(self, cer: float) -> Tuple[bool, Dict[str, Any]]:

        info = {"cer": float(cer), "reference_cer": self._reference_cer, "ph_stat": None, "burnin": False}

        # burn-in to set reference
        if self._reference_cer is None:
            self._cer_history.append(float(cer))
            info["burnin"] = True
            if len(self._cer_history) >= self.burn_in_windows:
                self._reference_cer = sum(self._cer_history) / len(self._cer_history)
                self._cer_history.clear()
                self._sum_m = 0.0
                self._min_sum_m = 0.0
            return False, info

        # PH update (no delta, fixed reference)
        dev = float(cer) - float(self._reference_cer)
        self._sum_m += dev
        self._min_sum_m = min(self._min_sum_m, self._sum_m)
        ph_stat = self._sum_m - self._min_sum_m
        info["reference_cer"] = self._reference_cer
        info["ph_stat"] = ph_stat

        perf_drift = ph_stat > self.lambda_ph
        if perf_drift:
            # reset accumulators ONLY (keep reference fixed as requested)
            self._sum_m = 0.0
            self._min_sum_m = 0.0

        return perf_drift, info


# Unseen event buffer manager to track unseen samples and support gating logic for IL triggers.
@dataclass
class NoveltyBufferManager:

    # thresholds for sample-based gating
    min_total_unseen_samples: int = 20
    min_unseen_samples_per_class: int = 5
    max_total_unseen_samples: int = 5000
    min_unseen_ratio_in_window: float = 0.03
    max_wait_windows_since_first_unseen: int = 6

    # state
    unseen_df: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    first_unseen_window: Optional[str] = None
    windows_since_first_unseen: int = 0

    # append new unseen samples to the buffer(either in prefix or target), return how many were added
    def append_to_buffer(self, df_new: pd.DataFrame) -> int:

        if df_new.empty:
            return 0

        old_size = len(self.unseen_df)

        self.unseen_df = pd.concat(
            [self.unseen_df, df_new.copy()],
            ignore_index=True
        )

        return len(self.unseen_df) - old_size

    # Add novel samples from the current window to the buffer based on the row mask,
    # Return added count, unseen ratio in window, and unseen count in window
    def add_unseen_samples(self, window_key: str, batch_df: pd.DataFrame, row_mask: List[bool]) -> Tuple[int, float, int]:

        n = int(len(batch_df))
        if n == 0:
            return 0, 0.0, 0

        # track first unseen window and windows since then for gating logic
        if self.first_unseen_window is None and any(row_mask):
            self.first_unseen_window = str(window_key)
            self.windows_since_first_unseen = 0

        if any(row_mask):
            self.windows_since_first_unseen += 1

        # select rows
        df_novel = batch_df.loc[row_mask].copy()
        novel_cnt = int(len(df_novel))
        novel_ratio = float(novel_cnt / n) if n > 0 else 0.0

        added = self.append_to_buffer(df_novel)
        return added, novel_ratio, novel_cnt

    # Count unseen events in the buffer
    # An activity is counted as unseen if it is not contained in known_events
    def count_unseen_events_in_buffer(self, known_events: Set[str]) -> Dict[str, int]:

        unseen_event_counts: Dict[str, int] = {}

        if self.unseen_df.empty:
            return unseen_event_counts

        for _, row in self.unseen_df.iterrows():
            events = str(row["prefix"]).split()
            events.append(str(row["next_act"]))

            for event in events:
                if event not in known_events:
                    unseen_event_counts[event] = unseen_event_counts.get(event, 0) + 1

        return unseen_event_counts

    # Check if buffer has reached max size
    def buffer_max_reached(self) -> bool:

        return len(self.unseen_df) >= self.max_total_unseen_samples

    '''
    # Check whether the minimum sample gate is satisfied.
    # The buffer must satisfy both the minimum total sample threshold
    # and the minimum per-class unseen-event threshold.
    def sample_gates_satisfied(self, known_events: Set[str]) -> bool:
        if len(self.unseen_df) < self.min_total_unseen_samples:
            return False

        unseen_event_counts = self.count_unseen_events_in_buffer(known_events)

        if not unseen_event_counts:
            return False

        return all(
            cnt >= self.min_unseen_samples_per_class
            for cnt in unseen_event_counts.values()
        )
    '''

    def min_buffer_size_satisfied(self) -> bool:
        return len(self.unseen_df) >= self.min_total_unseen_samples

    def eligible_class_signal(self, known_events: Set[str]) -> bool:
        unseen_event_counts = self.count_unseen_events_in_buffer(known_events)

        if not unseen_event_counts:
            return False

        return any(
            cnt >= self.min_unseen_samples_per_class
            for cnt in unseen_event_counts.values()
        )

    '''
    Determine whether to trigger IL training based on the AND combination of:Unseen Occurrence Gate
    AND Minimum Sample Gate
    AND (
        buffer max reached
        OR perf_drift
        OR high_ratio
        OR waited_long
    )
    '''
    '''
    def should_trigger_train(
            self,
            unseen: bool,
            perf_drift: bool,
            unseen_ratio_in_window: float,
            known_events: Set[str],
    ) -> Tuple[bool, List[str]]:

        reasons = []

        if not unseen:
            return False, reasons

        if not self.sample_gates_satisfied(known_events):
            return False, reasons

        buffer_full = self.buffer_max_reached()
        waited_long = self.windows_since_first_unseen >= self.max_wait_windows_since_first_unseen
        high_ratio = unseen_ratio_in_window >= self.min_unseen_ratio_in_window

        ok = buffer_full or perf_drift or high_ratio or waited_long

        if ok:
            if buffer_full:
                reasons.append("buffer_max_reached")
            if perf_drift:
                reasons.append("perf_drop")
            if high_ratio:
                reasons.append("high_unseen_ratio")
            if waited_long:
                reasons.append("max_wait_exceeded")

        return ok, reasons
    '''

    def should_trigger_train(
            self,
            unseen: bool,
            perf_drift: bool,
            unseen_ratio_in_window: float,
            known_events: Set[str],
    ) -> Tuple[bool, List[str]]:

        reasons = []

        if not unseen:
            return False, reasons

        if not self.min_buffer_size_satisfied():
            return False, reasons

        eligible_class = self.eligible_class_signal(known_events)
        buffer_full = self.buffer_max_reached()
        waited_long = self.windows_since_first_unseen >= self.max_wait_windows_since_first_unseen
        high_ratio = unseen_ratio_in_window >= self.min_unseen_ratio_in_window

        ok = eligible_class or perf_drift or high_ratio or waited_long or buffer_full

        if ok:
            if eligible_class:
                reasons.append("eligible_unseen_class")
            if perf_drift:
                reasons.append("perf_drop")
            if high_ratio:
                reasons.append("high_unseen_ratio")
            if waited_long:
                reasons.append("max_wait_exceeded")
            if buffer_full:
                reasons.append("buffer_max_reached")

        return ok, reasons

    def clear(self):
        self.unseen_df = pd.DataFrame()
        self.first_unseen_window = None
        self.windows_since_first_unseen = 0


# Main drift detector class that combines performance-based and unseen thresholds for IL triggers.
@dataclass
class DriftDetector:
    known_train_events: Set[str]
    ph: PageHinkleyDriftDetector
    buffer: NoveltyBufferManager

    # Update the known event set after model update
    def update_known_events(self, known_events: Set[str]):
        self.known_train_events = set(known_events)

    # Update method for each new window of data
    # Returns whether to trigger IL, the current unseen buffer, and info for logging/analysis
    def update(self, window_key: str, batch_df: pd.DataFrame, acc: float) -> Tuple[bool, pd.DataFrame, Dict[str, Any]]:

        n = int(len(batch_df))
        cer = 1.0 - float(acc)

        # unseen labels based on next_act
        next_acts = batch_df["next_act"].astype(str).tolist()
        win_labels = set(next_acts)
        unseen_labels = win_labels - self.known_train_events

        # unseen tokens based on prefix tokens
        unseen_tokens: Set[str] = set()

        # scan prefixes for unseen tokens and mark rows with unseen tokens in prefix
        prefix_has_unseen = []
        for p in batch_df["prefix"].astype(str).tolist():
            toks = p.split()
            has = any(t not in self.known_train_events for t in toks)
            prefix_has_unseen.append(has)
            if has:
                unseen_tokens.update([t for t in toks if t not in self.known_train_events])

        # unseen: either unseen label or unseen token in prefix
        unseen = (len(unseen_labels) > 0) or (len(unseen_tokens) > 0)

        # row is unseen if next_act unseen OR prefix has unseen token
        next_act_unseen_mask = batch_df["next_act"].astype(str).isin(unseen_labels).tolist()
        row_mask = [a or b for a, b in zip(next_act_unseen_mask, prefix_has_unseen)]

        added, novel_ratio, novel_cnt = self.buffer.add_unseen_samples(window_key, batch_df, row_mask)

        # perf monitoring independent
        perf_drift, perf_info = self.ph.update(cer)

        # AND gating decision
        trigger, trigger_reasons = self.buffer.should_trigger_train(
            unseen=unseen,
            perf_drift=perf_drift,
            unseen_ratio_in_window=novel_ratio,
            known_events=self.known_train_events,
        )

        buffer_unseen_event_counts = self.buffer.count_unseen_events_in_buffer(
            self.known_train_events
        )

        info = {
            "window_key": str(window_key),
            "n": n,
            "acc": float(acc),
            "cer": float(cer),

            "unseen": unseen,
            "unseen_labels": sorted(list(unseen_labels)),
            "unseen_count_in_window": int(novel_cnt),
            "unseen_ratio_in_window": float(novel_ratio),
            "buffer_added": int(added),
            "buffer_total": int(len(self.buffer.unseen_df)),
            "buffer_unseen_event_counts": buffer_unseen_event_counts,
            "buffer_max_reached": bool(self.buffer.buffer_max_reached()),

            "first_unseen_window": self.buffer.first_unseen_window,
            "windows_since_first_unseen": int(self.buffer.windows_since_first_unseen),

            "perf_drift": bool(perf_drift),
            "perf_info": perf_info,

            "trigger_train": bool(trigger),
            "trigger_reasons": trigger_reasons,
        }

        return trigger, self.buffer.unseen_df, info
