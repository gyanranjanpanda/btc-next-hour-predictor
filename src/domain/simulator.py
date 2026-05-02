from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
from scipy import stats as sp_stats

from .errors import SimulationError
from .models import Candle, Prediction

logger = logging.getLogger(__name__)

_REGIME_MULTIPLIERS = {
    "low": 1.01,
    "normal": 1.03,
    "high": 1.06,
}


@dataclass
class SimulationResult:
    """Simulation output with prediction and diagnostic metadata."""

    prediction: Prediction
    simulated_prices: np.ndarray
    fitted_mu: float
    fitted_sigma: float
    fitted_nu: float
    fitting_method: str
    volatility_regime: str
    sigma_multiplier: float


class GBMSimulator:
    """
    Geometric Brownian Motion simulator with adaptive volatility estimation.

    Primary path: GARCH(1,1) with Student-t innovations via `arch` library.
    Fallback: EWMA volatility with t-distribution MLE.

    Key correctness details:
    - Student-t innovations are variance-normalised: Z * sqrt((nu-2)/nu)
      so that Var(Z_normalised) = 1 regardless of nu.
    - GBM drift correction: mu - sigma^2/2 (Itô correction).
    - nu is floored at 3.0 to guarantee finite variance and prevent
      the normalisation factor from collapsing near nu=2.
    """

    def __init__(
        self,
        num_simulations: int = 10_000,
        volatility_lookback: int = 24,
        ewma_span: int = 12,
    ) -> None:
        self.num_simulations = num_simulations
        self.volatility_lookback = volatility_lookback
        self.ewma_span = ewma_span

    def _fit_volatility_and_df(
        self, log_returns: np.ndarray
    ) -> tuple[float, float, float, str]:
        """
        Returns (mu, sigma, nu, method) where:
        - mu: mean log return (drift)
        - sigma: conditional volatility estimate for next period
        - nu: Student-t degrees of freedom (>= 3.0)
        - method: "garch" or "ewma"
        """
        # GARCH needs at least ~50 observations to converge reliably
        if len(log_returns) >= 50:
            try:
                mu, sigma, nu = self._fit_with_garch(log_returns)
                return mu, sigma, nu, "garch"
            except Exception:
                logger.debug("GARCH fitting failed, using EWMA fallback")

        mu, sigma, nu = self._fit_with_ewma(log_returns)
        return mu, sigma, nu, "ewma"

    def _fit_with_garch(
        self, log_returns: np.ndarray
    ) -> tuple[float, float, float]:
        """GARCH(1,1) with Student-t innovations using the arch library."""
        from arch import arch_model

        # Scale to percentage for numerical stability
        scaled = log_returns * 100.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                scaled,
                vol="Garch",
                p=1,
                q=1,
                dist="t",
                mean="Constant",
                rescale=False,
            )
            fitted = model.fit(disp="off", show_warning=False)

        forecast = fitted.forecast(horizon=1)
        forecast_var_pct = forecast.variance.values[-1, 0]
        sigma = np.sqrt(forecast_var_pct) / 100.0

        mu = float(np.mean(log_returns))
        nu = float(fitted.params.get("nu", 5.0))
        # Floor at 3.0: finite variance requires nu > 2, and values near 2
        # make the normalisation factor sqrt((nu-2)/nu) collapse
        nu = float(np.clip(nu, 3.0, 30.0))

        return mu, sigma, nu

    def _fit_with_ewma(
        self, log_returns: np.ndarray
    ) -> tuple[float, float, float]:
        """
        EWMA volatility with scipy MLE for Student-t degrees of freedom.
        Provides volatility clustering via exponential weighting of recent
        squared returns.
        """
        mu = float(np.mean(log_returns))

        # EWMA variance: heavier weight on recent observations
        decay = 2.0 / (self.ewma_span + 1.0)
        n = len(log_returns)
        weights = np.array([(1 - decay) ** i for i in range(n - 1, -1, -1)])
        weights /= weights.sum()
        weighted_var = float(np.sum(weights * (log_returns - mu) ** 2))
        sigma = np.sqrt(weighted_var)

        # MLE fit for degrees of freedom
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nu_fit, _, _ = sp_stats.t.fit(log_returns, floc=mu, fscale=sigma)
                nu = float(np.clip(nu_fit, 3.0, 30.0))
        except Exception:
            nu = 5.0

        return mu, sigma, nu

    def _detect_volatility_regime(self, log_returns: np.ndarray) -> str:
        """
        Classify current volatility regime by comparing recent realized
        volatility against the full-history baseline.
        """
        if len(log_returns) < self.volatility_lookback * 2:
            return "normal"

        recent_vol = np.std(log_returns[-self.volatility_lookback:], ddof=1)
        baseline_vol = np.std(log_returns, ddof=1)

        if baseline_vol <= 1e-10:
            return "normal"

        ratio = recent_vol / baseline_vol

        if ratio < 0.7:
            return "low"
        elif ratio > 1.3:
            return "high"
        return "normal"

    def predict_next_candle(
        self,
        recent_candles: list[Candle],
    ) -> SimulationResult:
        """
        Generates a 95% prediction interval for the next hourly close.

        Uses regime-adaptive sigma scaling instead of a flat multiplier,
        producing tighter intervals in calm markets and wider ones during
        high-volatility regimes.
        """
        min_required = self.volatility_lookback + 1
        if len(recent_candles) < min_required:
            raise SimulationError(
                f"Need >= {min_required} candles, got {len(recent_candles)}"
            )

        closes = np.array([c.close_price for c in recent_candles])
        log_returns = np.diff(np.log(closes))

        # Feed the entire available return series to GARCH for better fitting,
        # but cap at 500 to avoid diluting with ancient regime data.
        fitting_returns = log_returns[-500:]
        mu, sigma, nu, fitting_method = self._fit_volatility_and_df(fitting_returns)

        # Safety floor: if sigma is degenerate, fall back to simple std
        if sigma <= 1e-10:
            sigma = float(np.std(log_returns[-self.volatility_lookback:], ddof=1))
        if sigma <= 1e-10:
            sigma = 1e-6

        raw_sigma = sigma

        # Regime-adaptive model-risk buffer instead of flat 1.05 multiplier.
        # Calm markets get minimal inflation; volatile markets get more.
        regime = self._detect_volatility_regime(fitting_returns)
        multiplier = _REGIME_MULTIPLIERS[regime]
        sigma *= multiplier

        current_price = closes[-1]
        dt = 1.0

        # Variance-normalised Student-t innovations
        # Raw t(nu) has Var = nu/(nu-2); scaling by sqrt((nu-2)/nu) gives Var = 1
        raw_z = sp_stats.t.rvs(df=nu, size=self.num_simulations)
        normalised_z = raw_z * np.sqrt((nu - 2.0) / nu)

        # GBM: S(t+dt) = S(t) * exp((mu - sigma²/2)*dt + sigma*sqrt(dt)*Z)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * normalised_z
        simulated_prices = current_price * np.exp(drift + diffusion)

        lower = float(np.percentile(simulated_prices, 2.5))
        upper = float(np.percentile(simulated_prices, 97.5))

        prediction = Prediction(
            timestamp=recent_candles[-1].timestamp,
            lower_bound=lower,
            upper_bound=upper,
            confidence_interval=0.95,
        )

        return SimulationResult(
            prediction=prediction,
            simulated_prices=simulated_prices,
            fitted_mu=mu,
            fitted_sigma=raw_sigma,
            fitted_nu=nu,
            fitting_method=fitting_method,
            volatility_regime=regime,
            sigma_multiplier=multiplier,
        )
