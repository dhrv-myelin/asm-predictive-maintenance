"""
drift/drift_detector.py
─────────────────────────
Detects feature distribution drift between a reference window
(training data) and the current window (recent production data).

Two complementary tests are used:
  PSI (Population Stability Index)
      Industry standard for monitoring. Bins both distributions,
      computes the divergence per bin.
      PSI < 0.1  → no significant drift
      PSI 0.1–0.2 → moderate drift (monitor)
      PSI > 0.2  → significant drift → retrain

  KS test (Kolmogorov-Smirnov)
      Non-parametric two-sample test. Sensitive to any distributional
      difference including shifts in mean, std, or shape.
      p-value < threshold → distributions differ significantly.

Both signals are aggregated into a single DriftReport.
The orchestrator reads DriftReport.drifted to decide whether to retrain.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from config.config_schema import MetricConfig
from data.data_loader import DriftDataset

logger = logging.getLogger(__name__)

_PSI_BINS = 10
_PSI_EPSILON = 1e-6   # avoid log(0)


@dataclass
class DriftReport:
    drifted              : bool
    drifted_features     : list[str]
    psi_scores           : dict[str, float]       # per feature
    ks_pvalues           : dict[str, float]        # per feature
    overall_drift_score  : float                   # mean PSI across all features
    reference_n          : int
    current_n            : int
    threshold_psi        : float
    threshold_ks_pvalue  : float
    drift_reason         : Optional[str] = None    # human-readable summary


class DriftDetector:
    """
    Parameters
    ----------
    config : MetricConfig — reads drift thresholds from config.training
    """

    def __init__(self, config: MetricConfig) -> None:
        self._config          = config
        self._psi_threshold   = config.training.drift_threshold_psi
        self._ks_threshold    = config.training.drift_threshold_ks_pvalue

    def check(self, drift_dataset: DriftDataset) -> DriftReport:
        """
        Run PSI and KS tests on every feature column.

        A feature is flagged as drifted if:
          PSI > psi_threshold  OR  KS p-value < ks_threshold

        Overall drifted=True if ANY feature is flagged.

        Returns a DriftReport regardless of whether drift was detected.
        """
        ref     = drift_dataset.reference_df
        cur     = drift_dataset.current_df
        features = drift_dataset.feature_names

        psi_scores     : dict[str, float] = {}
        ks_pvalues     : dict[str, float] = {}
        drifted_feats  : list[str]        = []

        for feat in features:
            if feat not in ref.columns or feat not in cur.columns:
                logger.debug("Drift: feature '%s' not in one of the windows; skipping.", feat)
                continue

            ref_vals = ref[feat].dropna().values.astype(float)
            cur_vals = cur[feat].dropna().values.astype(float)

            if len(ref_vals) < 5 or len(cur_vals) < 5:
                continue

            psi = self._compute_psi(ref_vals, cur_vals)
            ks_stat, ks_p = self._compute_ks(ref_vals, cur_vals)

            psi_scores[feat] = round(psi, 5)
            ks_pvalues[feat] = round(ks_p, 5)

            is_drifted = (psi > self._psi_threshold) or (ks_p < self._ks_threshold)
            if is_drifted:
                drifted_feats.append(feat)
                logger.info(
                    "Drift detected in %s/%s — feature '%s': PSI=%.3f, KS_p=%.4f",
                    self._config.station, self._config.metric, feat, psi, ks_p,
                )

        overall_psi = float(np.mean(list(psi_scores.values()))) if psi_scores else 0.0
        drifted     = len(drifted_feats) > 0

        reason: Optional[str] = None
        if drifted:
            top = sorted(
                [(f, psi_scores.get(f, 0.0)) for f in drifted_feats],
                key=lambda x: x[1], reverse=True,
            )
            reason = (
                f"{len(drifted_feats)} feature(s) drifted. "
                f"Highest PSI: {top[0][0]} = {top[0][1]:.3f}"
            )

        return DriftReport(
            drifted             = drifted,
            drifted_features    = drifted_feats,
            psi_scores          = psi_scores,
            ks_pvalues          = ks_pvalues,
            overall_drift_score = overall_psi,
            reference_n         = len(drift_dataset.reference_df),
            current_n           = len(drift_dataset.current_df),
            threshold_psi       = self._psi_threshold,
            threshold_ks_pvalue = self._ks_threshold,
            drift_reason        = reason,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Statistical tests
    # ──────────────────────────────────────────────────────────────────────

    def _compute_psi(self, reference: np.ndarray, current: np.ndarray) -> float:
        """
        Population Stability Index.
        Bins the reference distribution into _PSI_BINS equal-frequency bins,
        then measures how much the current distribution deviates.

        PSI = Σ (current% - reference%) × ln(current% / reference%)
        """
        # Use reference quantiles as bin edges
        quantiles = np.linspace(0, 100, _PSI_BINS + 1)
        bins      = np.unique(np.percentile(reference, quantiles))

        if len(bins) < 2:
            return 0.0

        ref_counts = np.histogram(reference, bins=bins)[0] + _PSI_EPSILON
        cur_counts = np.histogram(current,   bins=bins)[0] + _PSI_EPSILON

        ref_pct = ref_counts / ref_counts.sum()
        cur_pct = cur_counts / cur_counts.sum()

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(0.0, psi)

    def _compute_ks(
        self, reference: np.ndarray, current: np.ndarray
    ) -> tuple[float, float]:
        """
        Two-sample Kolmogorov-Smirnov test.
        Returns (statistic, p-value).
        p < threshold → distributions differ significantly.
        """
        stat, pvalue = scipy_stats.ks_2samp(reference, current)
        return float(stat), float(pvalue)
