"""Valid lower bounds for the project's clipped-log1p forecasting targets."""
import numpy as np

try:
    from .var_p import VAR
except ImportError:
    from var_p import VAR


def activity_floor(mu, sd, log_transform=True):
    """Raw zero in standardized clipped-log1p units; negative z-scores are valid."""
    mu, sd = np.asarray(mu), np.asarray(sd)
    if not log_transform:
        raise ValueError('This bound requires clipped-log1p target preprocessing')
    if (mu.ndim != 1 or not len(mu) or mu.shape != sd.shape or not np.isfinite(mu).all()
            or not np.isfinite(sd).all() or np.any(sd <= 0)):
        raise ValueError('Provide finite per-neuron means and positive scales')
    return np.asarray(-mu/sd, dtype=np.float64)


class BoundedForecast:
    """Apply the lower bound at each step BEFORE feeding predictions back."""
    def __init__(self, model, floor):
        floor = np.asarray(floor, dtype=np.float64)
        if model.intercept is None:
            raise RuntimeError('Fit the base model first')
        if floor.shape != model.intercept.shape or not np.isfinite(floor).all():
            raise ValueError('The floor must match the fitted neuron columns')
        self.base = model
        self.floor = floor.copy()
        self.P, self.intercept = model.P, model.intercept

    @property
    def parameter_count(self):
        return self.base.parameter_count

    def _history(self, history):
        return self.base._history(history)

    def predict(self, history):
        return np.maximum(self.base.predict(history), self.floor)

    rollout = VAR.rollout
