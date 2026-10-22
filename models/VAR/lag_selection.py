"""Common-target train/validation comparison of population VAR lag orders."""
import numpy as np

try:
    from .var_p import VAR
    from .evaluation import ForecastMetrics
except ImportError:
    from var_p import VAR
    from evaluation import ForecastMetrics


def history_batches(trials, P, context=15, horizon=1, batch_size=256):
    if (any(not isinstance(v, (int, np.integer)) or v < 1
            for v in (P, context, horizon, batch_size)) or P > context):
        raise ValueError('Positive integer settings and P <= context are required')
    for trial in trials:
        x = np.asarray(trial, dtype=np.float64)
        if x.ndim != 2 or not x.shape[1] or not np.isfinite(x).all():
            raise ValueError('Each trial must be finite [time, neurons]')
        for start in range(context, len(x)-horizon+1, batch_size):
            origins = np.arange(start, min(start+batch_size, len(x)-horizon+1))
            yield (x[origins[:, None]+np.arange(-P, 0)],
                   x[origins[:, None]+np.arange(horizon)])


def evaluate(model, trials, context=15, horizon=1, batch_size=256):
    if model.intercept is None:
        raise RuntimeError('Fit before evaluation')
    metrics = ForecastMetrics(horizon, len(model.intercept))
    for history, target in history_batches(trials, model.P, context, horizon, batch_size):
        metrics.update(model.rollout(history, horizon), target)
    result = metrics.result()
    result['rmse'] = np.sqrt(result['mse']).tolist()
    # Same macro-per-neuron definition/epsilon as models/common.py, using
    # float64 centered streaming moments rather than keeping all predictions.
    result['r2'] = (1-metrics.squared_error/(metrics.target_ss+1e-8)).mean(1).tolist()
    result['n_windows'] = metrics.count
    return result


def select_lags(train_trials, validation_trials, lags=(1, 2, 4), lambdas=(0.01, 0.1, 1),
                context=15, batch_size=256, verbose=True):
    """Select by one-step validation MSE; no test argument or test evaluation.

    Every P uses identical train AND validation targets (t >= context). Its own
    lagged feature matrix changes. Candidate-order breaks exact ties.
    """
    train_trials, validation_trials = list(train_trials), list(validation_trials)
    lags, lambdas = list(lags), list(lambdas)
    if (not lags or any(not isinstance(p, (int, np.integer)) or p < 1 for p in lags)
            or not isinstance(context, (int, np.integer)) or context < max(lags)):
        raise ValueError('Provide positive lag orders with context >= max(lags)')
    if not lambdas or any(not np.isfinite(v) or v <= 0 for v in lambdas):
        raise ValueError('Provide finite positive ridge penalties')
    grid, best_model, best_score = [], None, float('inf')
    for P in lags:
        if verbose:
            print(f'Accumulating training statistics for P={P}...', flush=True)
        statistics = VAR(P).statistics(train_trials, start=context)
        for lam in lambdas:
            model = VAR(P, lam).fit_statistics(statistics)
            training = evaluate(model, train_trials, context, batch_size=batch_size)
            validation = evaluate(model, validation_trials, context, batch_size=batch_size)
            score = validation['mse'][0]
            grid.append({'P': int(P), 'lambda': float(lam), 'parameter_count': model.parameter_count,
                         'fitted_training_pairs': model.n_train_pairs,
                         'train': training, 'validation': validation})
            if verbose:
                print(f'  P={P} lambda={lam:g} train MSE={training["mse"][0]:.6f} '
                      f'validation MSE={score:.6f}', flush=True)
            if score < best_score:
                best_model, best_score = model, score
        del statistics
    return best_model, grid
