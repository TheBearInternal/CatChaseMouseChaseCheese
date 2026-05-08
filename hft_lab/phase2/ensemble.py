"""Kalman-filter ensemble weighting and IC tracking for hft_lab Phase 2.

Components
----------
InformationCoefficient : Rolling Pearson correlation of signal predictions vs
                         actual outcomes, per signal, windowed to IC_WINDOW.
KalmanEnsemble         : Tracks a 6-element weight vector via a Kalman random-walk
                         model, updating when new IC observations arrive.
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
MIN_WEIGHT_FLOOR: float = 0.01        # minimum weight to prevent zeroing a signal


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
        self._history: Dict[str, deque[Tuple[float, float]]] = {
            name: deque(maxlen=config.ic_window) for name in SIGNAL_NAMES
        }

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
    """Tracks signal weight vector via a Kalman random-walk filter.

    State model
    -----------
    State transition : w_t = w_{t-1} + process_noise    (random walk)
    Observation      : z_t = H w_t + measurement_noise  (H = identity)

    The weight vector is updated whenever a new IC observation is available
    (after a trade closes and InformationCoefficient is updated).
    """

    def __init__(self) -> None:
        self._w: np.ndarray = np.full(N_SIGNALS, INITIAL_WEIGHT)
        self._P: np.ndarray = np.eye(N_SIGNALS) * INITIAL_COV_SCALE
        self._Q: np.ndarray = np.eye(N_SIGNALS) * config.kalman_process_noise
        self._R: float = config.kalman_measurement_noise

    def predict(self) -> None:
        """Kalman prediction step: advance covariance by process noise.

        Called once per trading session or periodically to allow weight
        uncertainty to grow before the next update.
        """
        self._P = self._P + self._Q
        logger.debug("KalmanEnsemble: prediction step applied")

    def update(self, ic_observation_vector: np.ndarray) -> None:
        """Kalman correction step using IC values as observations.

        Uses H = I (identity), so IC values are treated as direct noisy
        observations of the latent signal weights.

        Args:
            ic_observation_vector: 1-D array of length N_SIGNALS with current
                                   IC values in the order of SIGNAL_NAMES.
        """
        H = np.eye(N_SIGNALS)
        S = H @ self._P @ H.T + self._R * np.eye(N_SIGNALS)
        try:
            K = self._P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("KalmanEnsemble: singular S matrix — skipping update")
            return
        innovation = ic_observation_vector - H @ self._w
        self._w = self._w + K @ innovation
        self._P = (np.eye(N_SIGNALS) - K @ H) @ self._P
        logger.debug(
            "KalmanEnsemble updated | weights=" +
            " ".join(f"{SIGNAL_NAMES[i]}={self._w[i]:.4f}" for i in range(N_SIGNALS))
        )

    def get_weights(self) -> Dict[str, float]:
        """Return normalized, strictly-positive weight dict.

        All raw weights are floored at ``MIN_WEIGHT_FLOOR`` then normalized
        so they sum to 1.0.

        Returns:
            Dict mapping signal name to weight.
        """
        w_pos = np.maximum(self._w, MIN_WEIGHT_FLOOR)
        w_norm = w_pos / w_pos.sum()
        return {name: float(w_norm[i]) for i, name in enumerate(SIGNAL_NAMES)}

    def save_state(self) -> Dict[str, Any]:
        """Serialise weight vector for SlowBuffer persistence.

        Returns:
            Dict with ``weights`` and ``covariance`` lists.
        """
        return {
            "weights": self._w.tolist(),
            "covariance": self._P.tolist(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        """Restore weight vector from a previously saved SlowBuffer snapshot.

        Args:
            state: Dict produced by ``save_state``.
        """
        try:
            w = np.array(state["weights"], dtype=float)
            P = np.array(state["covariance"], dtype=float)
            if w.shape == (N_SIGNALS,) and P.shape == (N_SIGNALS, N_SIGNALS):
                self._w = w
                self._P = P
                logger.info("KalmanEnsemble: loaded weights from session state")
        except Exception as exc:
            logger.warning(f"KalmanEnsemble: could not load state — {exc}")


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
        logger.debug(
            f"Ensemble | regime={regime.value} raw={raw_score:+.4f} "
            f"sentiment={sentiment:+.3f} modified={modified_score:+.4f} "
            f"threshold={threshold:.2f} → {action} (conf={confidence:.4f})"
        )
        return decision

    @staticmethod
    def _apply_regime_priors(
        weights: Dict[str, float], regime: RegimeState
    ) -> Dict[str, float]:
        """Adjust weights based on which signals suit the current regime.

        Args:
            weights: Base weight dict from KalmanEnsemble.
            regime:  Current RegimeState.

        Returns:
            Renormalized weight dict.
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
            w = {k: 0.5 * v + 0.5 * equal for k, v in w.items()}

        total = sum(w.values())
        if total > 1e-8:
            w = {k: v / total for k, v in w.items()}
        return w
