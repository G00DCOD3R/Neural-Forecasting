"""Trial-safe recursive evaluation with GVAR-compatible prediction origins."""
import numpy as np


def forecast_batches(trials, context=15, horizon=5, batch_size=256):
    """Yield latest observed [batch, neurons] and targets [batch, horizon, neurons].

    A target origin t is valid only when x[t-context:t+horizon] fits inside
    one trial. VAR1 uses x[t-1], while the common context aligns origins with
    models needing more history. No full session-sized window tensor is stored.
    """
    if any(not isinstance(v, (int, np.integer)) or v < 1
           for v in (context, horizon, batch_size)):
        raise ValueError('context, horizon and batch_size must be positive integers')
    for trial in trials:
        x = np.asarray(trial, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError('Each trial must be a finite [time, neurons] array')
        for start in range(context, len(x) - horizon + 1, batch_size):
            origins = np.arange(start, min(start + batch_size, len(x) - horizon + 1))
            yield x[origins-1], x[origins[:, None] + np.arange(horizon)]


class ForecastMetrics:
    """Stream error and centered Pearson statistics per horizon and neuron.

    Batch-centered moments are merged to avoid subtracting large raw moments
    when activity has a constant offset. MSE weights individual observations.
    """
    def __init__(self, horizon, num_neurons):
        self.count = 0
        self.pred_mean = np.zeros((horizon, num_neurons))
        self.target_mean = np.zeros_like(self.pred_mean)
        self.pred_ss = np.zeros_like(self.pred_mean)
        self.target_ss = np.zeros_like(self.pred_mean)
        self.cross = np.zeros_like(self.pred_mean)
        self.squared_error = np.zeros_like(self.pred_mean)
        self.absolute_error = np.zeros_like(self.pred_mean)

    def update(self, prediction, target):
        prediction = np.asarray(prediction, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if (prediction.shape != target.shape or prediction.ndim != 3
                or prediction.shape[1:] != self.pred_mean.shape or len(target) == 0):
            raise ValueError('Expected matching nonempty [batch, horizon, neurons] arrays')
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError('Metric inputs must be finite')
        batch_count = len(target)
        total = self.count + batch_count
        pred_mean, target_mean = prediction.mean(0), target.mean(0)
        pc, tc = prediction - pred_mean, target - target_mean
        delta_pred, delta_target = pred_mean - self.pred_mean, target_mean - self.target_mean
        weight = self.count * batch_count / total
        self.pred_ss += (pc**2).sum(0) + delta_pred**2 * weight
        self.target_ss += (tc**2).sum(0) + delta_target**2 * weight
        self.cross += (pc*tc).sum(0) + delta_pred*delta_target * weight
        self.pred_mean += delta_pred * (batch_count / total)
        self.target_mean += delta_target * (batch_count / total)
        error = prediction - target
        self.squared_error += (error**2).sum(0)
        self.absolute_error += abs(error).sum(0)
        self.count = total

    def result(self):
        if not self.count:
            raise ValueError('No forecast windows: at least one trial must contain context + horizon samples')
        # Match GVAR's epsilon-denominator convention. Undefined correlations
        # for constant neuron pairs contribute zero to the across-neuron mean.
        correlation = self.cross / (np.sqrt(self.pred_ss*self.target_ss) + 1e-8)
        return {'mse': (self.squared_error / self.count).mean(1).tolist(),
                'mae': (self.absolute_error / self.count).mean(1).tolist(),
                'pearson': np.clip(correlation, -1, 1).mean(1).tolist(),
                'mse_per_neuron_h1': (self.squared_error[0] / self.count).tolist()}


def evaluate_recursive(model, trials, context=15, horizon=5, batch_size=256, mean_prediction=None):
    """VAR and copy-last metrics on identical origins/targets at every horizon."""
    if model.coefficients is None:
        raise RuntimeError('Fit the model before evaluating')
    if any(not isinstance(v, (int, np.integer)) or v < 1
           for v in (context, horizon, batch_size)):
        raise ValueError('context, horizon and batch_size must be positive integers')
    num_neurons = len(model.intercept)
    metrics = {name: ForecastMetrics(horizon, num_neurons) for name in ('var', 'copy_last')}
    if mean_prediction is not None:
        mean_prediction = np.asarray(mean_prediction, dtype=np.float64)
        if mean_prediction.shape != (num_neurons,) or not np.isfinite(mean_prediction).all():
            raise ValueError('mean_prediction must be one finite value per neuron')
        metrics['mean'] = ForecastMetrics(horizon, num_neurons)
    for previous, target in forecast_batches(trials, context, horizon, batch_size):
        metrics['var'].update(model.rollout(previous, horizon), target)
        # Copy-last repeats the initial observed value at every future step.
        metrics['copy_last'].update(np.broadcast_to(previous[:, None], target.shape), target)
        if mean_prediction is not None:
            metrics['mean'].update(np.broadcast_to(mean_prediction, target.shape), target)
    return {**{name: accumulator.result() for name, accumulator in metrics.items()},
            'n_windows': metrics['var'].count, 'context': context, 'horizon': horizon}
