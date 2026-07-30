"""Kalman-filter ensemble weighting and IC tracking for hft_lab Phase 2.

Components
----------
InformationCoefficient : Rolling Pearson correlation of signal predictions vs
                         actual outcomes, per signal, windowed to IC_WINDOW.
KalmanEnsemble         : Maintains four independent Kalman weight vectors —
                         one per RegimeState — so each regime accumulates its
                         own signal performance history without cross-regime
                         dilution.
EnsembleDecision       : Combines weighted signal scores, regime-based weight
                         priors, and news sentiment to produce a final trading
                         decision with a configurable confidence threshold.
Decision               : NamedTuple carrying all decision metadata.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr

from config import config
from logger import get_logger
from phase2.regime import RegimeState
from phase2.signals import SIGNAL_NAMES

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_SIGNALS: int = len(SIGNAL_NAMES)
INITIAL_WEIGHT: float = 1.0 / N_SIGNALS
INITIAL_COV_SCALE: float = 0.1
REGIME_BOOST_FACTOR: float = 0.20     # fractional weight boost in aligned regime
SENTIMENT_MODIFIER_SCALE: float = 0.20
L1_NORM_TOLERANCE: float = 1e-6       # startup assertion tolerance on sum(|w|)

# Weight-semantics version stamped into saved state.  Files written before
# signed weights existed carry no such key: their raw vectors could already
# hold negatives, but those were inert because get_weights() floored every
# weight at +0.01.  Restoring one now makes those negatives ACTIVE fade
# weights, which can change trade direction — so the load path says so loudly.
WEIGHTS_FORMAT: str = "signed_l1_v2"


def _l1_normalize(w: "np.ndarray", cap: Optional[float] = None) -> "np.ndarray":
    """L1-normalize a signed weight vector so sum(|w|) == 1, retaining sign.

    Unlike sum-normalization, the denominator sum(|w|) can never be negative
    or cancel to ~zero for a non-degenerate vector, so the vector's signs are
    never inverted.  A degenerate (all ~zero / non-finite) vector falls back
    to uniform positive weights.

    Args:
        w:   Raw signed weight vector.
        cap: Optional per-element magnitude ceiling; elements with |w| above
             it are clipped (sign preserved) and the vector re-normalized,
             iterating until no element violates the cap.

    Returns:
        Signed vector with sum(|w|) == 1.
    """
    w = np.asarray(w, dtype=float)
    n = len(w)
    total = float(np.sum(np.abs(w)))
    if not np.isfinite(total) or total < 1e-12:
        return np.full(n, 1.0 / n)
    w = w / total
    if cap is None:
        return w

    signs = np.sign(w)
    signs[signs == 0.0] = 1.0
    m = np.abs(w)

    # A cap below 1/n cannot satisfy sum(|w|) == 1; uniform magnitude is the
    # closest feasible point.
    if cap * n <= 1.0 + 1e-12:
        return signs * (1.0 / n)

    # Water-filling: pin violators at the cap and redistribute the remaining
    # budget across the rest, repeating since redistribution can push a
    # previously-compliant element over.  Terminates in <= n rounds because
    # each round pins at least one more element.
    for _ in range(n):
        over = m > cap + 1e-15
        if not over.any():
            break
        free = ~over
        m[over] = cap
        budget = 1.0 - cap * float(over.sum())
        free_sum = float(m[free].sum())
        if free_sum > 1e-15:
            m[free] = m[free] * (budget / free_sum)
        elif free.any():
            m[free] = budget / float(free.sum())
    return signs * m


# ---------------------------------------------------------------------------
# Decision namedtuple
# ---------------------------------------------------------------------------


class Decision(NamedTuple):
    """Complete metadata for one ensemble trading decision.

    Attributes
    ----------
    action        : "LONG", "SHORT", or "HOLD".
    confidence    : Absolute value of modified_score.
    regime        : RegimeState at time of decision.
    signal_scores : Dict of raw signal scores used.
    weights_used  : Dict of weights applied to each signal.
    raw_score     : Dot product of weights and signal scores before sentiment.
    """

    action: str
    confidence: float
    regime: RegimeState
    signal_scores: Dict[str, float]
    weights_used: Dict[str, float]
    raw_score: float


# ---------------------------------------------------------------------------
# InformationCoefficient
# ---------------------------------------------------------------------------


class InformationCoefficient:
    """Rolling Pearson correlation tracker between signal predictions and outcomes.

    One deque of (prediction, actual_return) tuples is maintained per signal.
    IC values are re-computed on each ``get_ic`` call from the stored history.
    """

    def __init__(self) -> None:
        if config.is_crypto():
            ic_window = config.ic_window_crypto
        elif config.is_forex():
            ic_window = config.ic_window_forex
        else:
            ic_window = config.ic_window_equity
        self._history: Dict[str, deque[Tuple[float, float]]] = {
            name: deque(maxlen=ic_window) for name in SIGNAL_NAMES
        }
        logger.debug(
            f"InformationCoefficient | window={ic_window} "
            f"market={config.market_type}"
        )

    def update(self, signal_name: str, prediction: float, actual_return: float) -> None:
        """Record one (prediction, outcome) pair for a signal.

        Args:
            signal_name:   Signal identifier (must be in SIGNAL_NAMES).
            prediction:    Score emitted by the signal at entry.
            actual_return: Realised return over the trade duration.
        """
        if signal_name in self._history:
            self._history[signal_name].append((prediction, actual_return))

    def get_ic(self, signal_name: str) -> float:
        """Return the Pearson IC for one signal, or 0.0 if insufficient data.

        Args:
            signal_name: Signal identifier.

        Returns:
            Pearson r in [-1.0, +1.0], or 0.0 if fewer than 4 samples exist.
        """
        if signal_name not in self._history:
            return 0.0
        hist = list(self._history[signal_name])
        if len(hist) < 4:
            return 0.0
        preds = np.array([h[0] for h in hist])
        actuals = np.array([h[1] for h in hist])
        if np.std(preds) < 1e-10 or np.std(actuals) < 1e-10:
            return 0.0
        try:
            r, _ = pearsonr(preds, actuals)
            return float(np.clip(r, -1.0, 1.0)) if not math.isnan(r) else 0.0
        except Exception:
            return 0.0

    def get_all_ics(self) -> Dict[str, float]:
        """Return a dict of IC values for every signal.

        Returns:
            Dict mapping signal name to IC in [-1.0, +1.0].
        """
        return {name: self.get_ic(name) for name in SIGNAL_NAMES}


# ---------------------------------------------------------------------------
# KalmanEnsemble
# ---------------------------------------------------------------------------


class KalmanEnsemble:
    """Four independent Kalman weight vectors — one per RegimeState.

    State model (per regime)
    ------------------------
    State transition : w_t = w_{t-1} + process_noise    (random walk)
    Observation      : z_t = H w_t + measurement_noise  (H = identity)

    Each regime's vector is updated only when a trade closes in that regime,
    so TRENDING weights learn only from trending-market outcomes, and so on.
    Switching the active regime logs at DEBUG so operators can trace which
    vector is driving decisions at any moment.
    """

    _REGIME_SAVE_KEY: Dict[RegimeState, str] = {
        RegimeState.TRENDING: "trending",
        RegimeState.MEAN_REVERTING: "mean_reverting",
        RegimeState.RANDOM_WALK: "random_walk",
        RegimeState.AMBIGUOUS: "ambiguous",
    }

    def __init__(self) -> None:
        self._weights: Dict[RegimeState, np.ndarray] = {
            r: np.full(N_SIGNALS, INITIAL_WEIGHT) for r in RegimeState
        }
        self._covariance: Dict[RegimeState, np.ndarray] = {
            r: np.eye(N_SIGNALS) * INITIAL_COV_SCALE for r in RegimeState
        }
        self._Q: np.ndarray = np.eye(N_SIGNALS) * config.kalman_process_noise
        self._R: float = config.kalman_measurement_noise
        self._active_regime: Optional[RegimeState] = None
        # L1-normalized shrinkage target for the net-bias cap, and provenance
        # so the warm start knows whether it is refining or initializing.
        self._prior_weights: Dict[RegimeState, np.ndarray] = {
            r: np.full(N_SIGNALS, INITIAL_WEIGHT) for r in RegimeState
        }
        self.priors_source: str = "uniform"
        self._net_cap_logged: Dict[RegimeState, bool] = {r: False for r in RegimeState}

    def load_study_priors(self, path: str) -> bool:
        """Seed each regime's weights and covariance from the offline study.

        Weights come from ``signal_priors.json`` (mean test-split IC per
        signal, L1-normalized).  Initial covariance is scaled by how
        well-evidenced each weight is — more observations and larger |t| give
        a tighter prior, so the filter moves off it more slowly::

            confidence = mean|t| x min(1, n / 1000)
            variance   = INITIAL_COV_SCALE / (1 + confidence)

        Args:
            path: Path to signal_priors.json.

        Returns:
            ``True`` when at least one regime was seeded.
        """
        import json
        import os

        if not os.path.exists(path):
            logger.info(
                f"Kalman priors | no study file at {path} — starting at uniform 1/6"
            )
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as exc:
            logger.warning(f"Kalman priors | unreadable ({exc!r}) — using uniform")
            return False

        regimes = payload.get("regimes") or {}
        meta = payload.get("_meta") or {}
        seeded = 0
        for regime in RegimeState:
            entry = regimes.get(regime.value)
            if not entry:
                continue
            weights = entry.get("weights") or {}
            stats = entry.get("stats") or {}
            try:
                vec = np.array(
                    [float(weights[n]) for n in SIGNAL_NAMES], dtype=float
                )
            except (KeyError, TypeError, ValueError):
                logger.warning(
                    f"Kalman priors | {regime.value} malformed — skipping"
                )
                continue
            if not np.isfinite(vec).all():
                logger.warning(
                    f"Kalman priors | {regime.value} non-finite — skipping"
                )
                continue

            vec = _l1_normalize(vec, cap=config.max_signal_weight)
            self._weights[regime] = vec.copy()
            self._prior_weights[regime] = vec.copy()

            variances = []
            for name in SIGNAL_NAMES:
                st = stats.get(name) or {}
                n_obs = float(st.get("n", 0) or 0)
                mean_t = float(st.get("mean_abs_t", 0.0) or 0.0)
                confidence = mean_t * min(1.0, n_obs / 1000.0)
                variances.append(INITIAL_COV_SCALE / (1.0 + max(confidence, 0.0)))
            self._covariance[regime] = np.diag(variances)
            seeded += 1

            detail = " ".join(
                f"{SIGNAL_NAMES[i]}={vec[i]:+.4f}" for i in range(N_SIGNALS)
            )
            tot_n = sum(int((stats.get(n) or {}).get("n", 0)) for n in SIGNAL_NAMES)
            avg_t = float(np.mean([
                float((stats.get(n) or {}).get("mean_abs_t", 0.0))
                for n in SIGNAL_NAMES
            ]))
            logger.info(
                f"Kalman priors | loaded from study | {regime.value}: {detail} "
                f"(n={tot_n}, mean|t|={avg_t:.2f})"
            )

        if seeded:
            self.priors_source = "study"
            logger.info(
                f"Kalman priors | seeded {seeded}/4 regimes from "
                f"{meta.get('horizon_minutes', '?')}-minute study horizon "
                f"(rs source: {meta.get('relative_strength_source', '?')})"
            )
        return seeded > 0

    def _apply_net_cap(
        self, w: np.ndarray, regime: RegimeState
    ) -> np.ndarray:
        """Shrink *w* toward the regime's prior until |sum(w)| <= max_net_weight.

        A vector whose mass is almost entirely one-signed makes the ensemble a
        near-pure inverter (or amplifier) of whatever the signals say.  Blending
        toward the prior pulls the net back without discarding the learned
        per-signal structure.  Bisection is used because the L1 renormalization
        after each blend makes the relationship non-linear.
        """
        cap = config.max_net_weight
        if cap <= 0.0 or abs(float(np.sum(w))) <= cap:
            self._net_cap_logged[regime] = False
            return w

        prior = _l1_normalize(
            self._prior_weights[regime], cap=config.max_signal_weight
        )
        if abs(float(np.sum(prior))) > cap:
            # Even the prior is more one-sided than the cap; nothing to shrink
            # toward, so leave the vector alone rather than distort it.
            if not self._net_cap_logged[regime]:
                logger.warning(
                    f"Net-weight cap | {regime.value} | net="
                    f"{float(np.sum(w)):+.4f} exceeds {cap:.2f} but the prior "
                    f"(net={float(np.sum(prior)):+.4f}) does too — not shrinking"
                )
                self._net_cap_logged[regime] = True
            return w

        before = float(np.sum(w))
        lo, hi = 0.0, 1.0
        best = _l1_normalize(prior, cap=config.max_signal_weight)
        for _ in range(40):
            mid = (lo + hi) / 2.0
            cand = _l1_normalize(
                (1.0 - mid) * w + mid * prior, cap=config.max_signal_weight
            )
            if abs(float(np.sum(cand))) <= cap:
                best, hi = cand, mid
            else:
                lo = mid
        if not self._net_cap_logged[regime]:
            logger.warning(
                f"Net-weight cap | {regime.value} | net={before:+.4f} -> "
                f"{float(np.sum(best)):+.4f} (cap {cap:.2f}, shrunk {hi:.0%} "
                "toward study prior)"
            )
            self._net_cap_logged[regime] = True
        return best

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_log_switch(self, regime: RegimeState) -> None:
        """Log at DEBUG whenever the active regime vector changes."""
        if regime == self._active_regime:
            return
        w_norm = _l1_normalize(self._weights[regime], cap=config.max_signal_weight)
        w_str = "[" + " ".join(f"{v:+.4f}" for v in w_norm) + "]"
        logger.debug(
            f"Kalman | switching to {regime.value} weights | w={w_str}"
        )
        self._active_regime = regime

    # ------------------------------------------------------------------
    # Kalman steps
    # ------------------------------------------------------------------

    def predict(self, regime: Optional[RegimeState] = None) -> None:
        """Kalman prediction step: advance covariance by process noise.

        Args:
            regime: Regime whose covariance to advance.  Pass ``None`` (the
                    default) to advance all four regimes — used at session open
                    before the first regime classification runs.
        """
        targets = list(RegimeState) if regime is None else [regime]
        for r in targets:
            self._covariance[r] = self._covariance[r] + self._Q
        label = regime.value if regime is not None else "all regimes"
        logger.debug(f"Kalman | predict step applied to {label}")

    def update(self, ic_observation_vector: np.ndarray, regime: RegimeState) -> None:
        """Kalman correction step for *regime*'s weight vector only.

        Uses H = I (identity), so IC values are direct noisy observations of
        the latent per-regime signal weights.

        Args:
            ic_observation_vector: 1-D array of length N_SIGNALS with current
                                   IC values in SIGNAL_NAMES order.
            regime: The regime whose vector to update.
        """
        self._maybe_log_switch(regime)
        w = self._weights[regime]
        P = self._covariance[regime]

        H = np.eye(N_SIGNALS)
        S = H @ P @ H.T + self._R * np.eye(N_SIGNALS)
        try:
            K = P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning(
                f"KalmanEnsemble: singular S matrix for {regime.value} — skipping update"
            )
            return
        innovation = ic_observation_vector - H @ w
        self._weights[regime] = w + K @ innovation
        self._covariance[regime] = (np.eye(N_SIGNALS) - K @ H) @ P
        logger.debug(
            f"Kalman updated [{regime.value}] | weights=" +
            " ".join(
                f"{SIGNAL_NAMES[i]}={self._weights[regime][i]:.4f}"
                for i in range(N_SIGNALS)
            )
        )

    def get_weights(self, regime: RegimeState) -> Dict[str, float]:
        """Return the L1-normalized SIGNED weight dict for *regime*.

        Negative weights survive intact — a negative weight means "fade this
        signal", which the Kalman filter learns when a signal's IC is
        consistently negative.  The vector is L1-normalized (sum of absolute
        weights == 1.0) so signs are never inverted by a near-zero or
        negative plain sum, and each |weight| is capped at
        ``config.max_signal_weight`` to stop one signal dominating.

        Args:
            regime: The regime whose weight vector to use.

        Returns:
            Dict mapping signal name to signed weight with sum(|w|) == 1.0.
        """
        self._maybe_log_switch(regime)
        w_norm = _l1_normalize(self._weights[regime], cap=config.max_signal_weight)
        w_norm = self._apply_net_cap(w_norm, regime)
        return {name: float(w_norm[i]) for i, name in enumerate(SIGNAL_NAMES)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(self) -> Dict[str, Any]:
        """Serialise all four regime weight vectors for SlowBuffer persistence.

        Returns:
            Dict with keys ``kalman_weights_<regime>`` and
            ``kalman_covariance_<regime>`` for each RegimeState.
        """
        state: Dict[str, Any] = {"weights_format": WEIGHTS_FORMAT}
        for regime, key in self._REGIME_SAVE_KEY.items():
            state[f"kalman_weights_{key}"] = self._weights[regime].tolist()
            state[f"kalman_covariance_{key}"] = self._covariance[regime].tolist()
            norm = _l1_normalize(
                self._weights[regime], cap=config.max_signal_weight
            )
            state[f"kalman_weights_normalized_{key}"] = norm.tolist()
            logger.info(
                f"Kalman persist [{regime.value}] | " + " ".join(
                    f"{SIGNAL_NAMES[i]}={norm[i]:+.4f}"
                    for i in range(N_SIGNALS)
                )
            )
        return state

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore weight vectors from a previously saved SlowBuffer snapshot.

        Handles both the current four-vector format and the legacy single-vector
        format (``weights`` / ``covariance`` keys) by broadcasting the legacy
        vector to all regimes so old session files are forward-compatible.

        Args:
            state: Dict produced by ``save_state`` or its legacy predecessor.
        """
        # Legacy single-vector format → broadcast to all regimes
        if "weights" in state and "covariance" in state:
            try:
                w = np.array(state["weights"], dtype=float)
                P = np.array(state["covariance"], dtype=float)
                if (
                    w.shape == (N_SIGNALS,)
                    and P.shape == (N_SIGNALS, N_SIGNALS)
                    and np.isfinite(w).all()
                    and np.isfinite(P).all()
                ):
                    for r in RegimeState:
                        self._weights[r] = w.copy()
                        self._covariance[r] = P.copy()
                    logger.info(
                        "KalmanEnsemble: upgraded legacy single-vector state "
                        "→ broadcast to all 4 regime vectors"
                    )
                    self._verify_loaded_state()
                    return
            except Exception:
                pass

        # Current four-vector format
        loaded = 0
        for regime, key in self._REGIME_SAVE_KEY.items():
            try:
                w = np.array(state[f"kalman_weights_{key}"], dtype=float)
                P = np.array(state[f"kalman_covariance_{key}"], dtype=float)
                if w.shape == (N_SIGNALS,) and P.shape == (N_SIGNALS, N_SIGNALS):
                    if not (np.isfinite(w).all() and np.isfinite(P).all()):
                        logger.error(
                            f"KalmanEnsemble: non-finite state for "
                            f"{regime.value} — keeping that regime at priors"
                        )
                        continue
                    self._weights[regime] = w
                    self._covariance[regime] = P
                    loaded += 1
            except Exception:
                pass  # missing regime keeps its equal-weight default

        if loaded > 0:
            logger.info(
                f"KalmanEnsemble: loaded {loaded}/4 regime weight vectors "
                "from session state"
            )
            self.priors_source = "restored"
            self._warn_if_legacy_semantics(state)
            self._verify_loaded_state()
        else:
            logger.warning(
                "KalmanEnsemble: no valid vectors found in state — "
                "all regimes start at equal weights"
            )

    def _warn_if_legacy_semantics(self, state: Dict[str, Any]) -> None:
        """Warn when a pre-signed-weights state file carries negative weights.

        Such files were written when ``get_weights()`` floored everything at
        +0.01, so any stored negative was inert.  Under signed weights the same
        file activates them as fade weights, which can flip trade direction on
        the first tick after upgrading.  The weights are kept (learned negative
        ICs are real information), but the change is surfaced explicitly rather
        than happening silently.
        """
        if state.get("weights_format") == WEIGHTS_FORMAT:
            return
        negatives = {
            regime.value: [
                SIGNAL_NAMES[i]
                for i in range(N_SIGNALS)
                if self._weights[regime][i] < 0.0
            ]
            for regime in RegimeState
        }
        flagged = {r: sigs for r, sigs in negatives.items() if sigs}
        if not flagged:
            return
        logger.warning(
            "KalmanEnsemble: session state predates signed weights "
            f"(no '{WEIGHTS_FORMAT}' stamp). Negative weights that were "
            "previously floored to +0.01 are now ACTIVE fade weights and may "
            "reverse trade direction: "
            + "; ".join(f"{r}: {', '.join(s)}" for r, s in flagged.items())
            + ". Delete the state file to relearn from priors if unintended."
        )

    def _verify_loaded_state(self) -> None:
        """Startup check: every regime's vector is finite and L1-normalizes.

        Deliberately does NOT raise.  ``load_state`` is called from
        ``SessionManager._session_start``, which is not exception-guarded, and
        the engine's shutdown handler unconditionally re-saves state — so
        raising here would crash startup AND let the shutdown path overwrite a
        good ``session_state.json`` with half-loaded weights.  A violated
        invariant therefore logs at ERROR and resets that regime to priors,
        which is recoverable.  (A bare ``assert`` would also be stripped
        entirely under ``python -O``, silently removing the check.)

        Logs the full signed normalized vector per regime at INFO so negative
        (fade) weights are visible immediately after restore.
        """
        for regime in RegimeState:
            w = self._weights[regime]
            if not np.isfinite(w).all():
                logger.error(
                    f"Kalman weights for {regime.value} are non-finite after "
                    "load — resetting this regime to equal-weight priors"
                )
                self._weights[regime] = np.full(N_SIGNALS, INITIAL_WEIGHT)
                self._covariance[regime] = np.eye(N_SIGNALS) * INITIAL_COV_SCALE
                w = self._weights[regime]
            norm = _l1_normalize(w, cap=config.max_signal_weight)
            l1 = float(np.sum(np.abs(norm)))
            if abs(l1 - 1.0) > L1_NORM_TOLERANCE:
                logger.error(
                    f"L1 normalization invariant violated for {regime.value}: "
                    f"sum(|w|)={l1:.9f} — resetting this regime to priors"
                )
                self._weights[regime] = np.full(N_SIGNALS, INITIAL_WEIGHT)
                self._covariance[regime] = np.eye(N_SIGNALS) * INITIAL_COV_SCALE
                norm = _l1_normalize(
                    self._weights[regime], cap=config.max_signal_weight
                )
            logger.info(
                f"Kalman loaded [{regime.value}] | " + " ".join(
                    f"{SIGNAL_NAMES[i]}={norm[i]:+.4f}"
                    for i in range(N_SIGNALS)
                )
            )


# ---------------------------------------------------------------------------
# EnsembleDecision
# ---------------------------------------------------------------------------


class EnsembleDecision:
    """Combines weighted signal scores, regime priors, and sentiment into a Decision.

    Regime-based weight priors
    --------------------------
    TRENDING      : MACD and RSI boosted by 20 %, then renormalized.
    MEAN_REVERTING: Bollinger and VWAP boosted by 20 %, then renormalized.
    RANDOM_WALK   : All weights blended 50/50 toward equal weighting (entropy
                    maximization — no signal is trusted in random walk).
    """

    def __init__(self) -> None:
        self._prev_raw: float = 0.0
        self._prev_action: str = ""

    def decide(
        self,
        signal_scores: Dict[str, float],
        weights: Dict[str, float],
        regime: RegimeState,
        sentiment: float,
    ) -> Decision:
        """Produce a trading decision from all available information.

        Steps
        -----
        1. Apply regime-based weight priors.
        2. Compute raw_score = dot(weights, scores).
        3. Apply sentiment modifier.
        4. Select threshold based on regime.
        5. Return LONG / SHORT / HOLD with confidence.

        Args:
            signal_scores: Dict of signal scores from ``SignalEngine``.
            weights:       Dict of Kalman weights from ``KalmanEnsemble``.
            regime:        Current ``RegimeState``.
            sentiment:     Float in [-1, +1] from ``SentimentAnalyzer``.

        Returns:
            ``Decision`` namedtuple.
        """
        adjusted = self._apply_regime_priors(weights, regime)
        score_arr = np.array([signal_scores.get(n, 0.0) for n in SIGNAL_NAMES])
        weight_arr = np.array([adjusted[n] for n in SIGNAL_NAMES])

        raw_score = float(np.dot(weight_arr, score_arr))
        modified_score = raw_score * (1.0 + sentiment * SENTIMENT_MODIFIER_SCALE)

        if regime == RegimeState.RANDOM_WALK:
            threshold = config.ensemble_threshold_random
        elif regime == RegimeState.AMBIGUOUS:
            threshold = config.ensemble_threshold_ambiguous
        else:
            threshold = config.ensemble_threshold

        confidence = abs(modified_score)
        if confidence >= threshold:
            action = "LONG" if modified_score > 0 else "SHORT"
        else:
            action = "HOLD"

        decision = Decision(
            action=action,
            confidence=confidence,
            regime=regime,
            signal_scores=dict(signal_scores),
            weights_used=adjusted,
            raw_score=raw_score,
        )
        if abs(raw_score - self._prev_raw) > 0.001 or action != self._prev_action:
            logger.debug(
                f"Ensemble | regime={regime.value} raw={raw_score:+.4f} "
                f"sentiment={sentiment:+.3f} modified={modified_score:+.4f} "
                f"threshold={threshold:.2f} → {action} (conf={confidence:.4f}) "
                f"| net_weight={float(np.sum(weight_arr)):+.4f}"
            )
            self._prev_raw = raw_score
            self._prev_action = action
        return decision

    @staticmethod
    def _apply_regime_priors(
        weights: Dict[str, float], regime: RegimeState
    ) -> Dict[str, float]:
        """Adjust weights based on which signals suit the current regime.

        Sign-preserving throughout: boosting a negative (fade) weight
        increases its magnitude rather than pulling it toward positive, and
        the RANDOM_WALK entropy blend shrinks toward uniform *magnitude*
        carrying each weight's own sign.  Blending toward a positive uniform
        prior would flip any weight whose magnitude sits below ``equal``,
        destroying exactly the fade information the Kalman filter learned.

        Args:
            weights: Base signed weight dict from KalmanEnsemble.
            regime:  Current RegimeState.

        Returns:
            L1-renormalized signed weight dict (sum of |weights| == 1.0).
        """
        w = dict(weights)

        if regime == RegimeState.TRENDING:
            w["macd"] *= (1.0 + REGIME_BOOST_FACTOR)
            w["rsi"] *= (1.0 + REGIME_BOOST_FACTOR)
        elif regime == RegimeState.MEAN_REVERTING:
            w["bollinger"] *= (1.0 + REGIME_BOOST_FACTOR)
            w["vwap"] *= (1.0 + REGIME_BOOST_FACTOR)
        elif regime == RegimeState.RANDOM_WALK:
            equal = 1.0 / N_SIGNALS
            w = {
                k: 0.5 * v + 0.5 * math.copysign(equal, v if v != 0.0 else 1.0)
                for k, v in w.items()
            }

        vec = _l1_normalize(
            np.array([w[name] for name in SIGNAL_NAMES], dtype=float),
            cap=config.max_signal_weight,
        )
        return {name: float(vec[i]) for i, name in enumerate(SIGNAL_NAMES)}
