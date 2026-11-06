"""One-step evaluation and validation-only ridge selection for VAR1."""
import numpy as np

try:
    from .var import VAR1
except ImportError:
    from var import VAR1


def evaluate_one_step(model, trials, context=1):
    """Score VAR and copy-last on the same within-trial target rows.

    A context of 15 can later align the first prediction time with GVAR.
    VAR1 still uses only the latest observed population vector.
    """
    if not isinstance(context, (int, np.integer)) or context < 1:
        raise ValueError('context must be a positive integer')
    squared_error = copy_error = 0.0
    observations = pairs = 0
    for trial in trials:
        x = np.asarray(trial, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError('Each trial must be a finite [time, neurons] array')
        if len(x) <= context:
            continue
        previous, target = x[context-1:-1], x[context:]
        prediction = model.predict(previous)
        squared_error += np.sum((prediction - target)**2)
        copy_error += np.sum((previous - target)**2)
        observations += target.size
        pairs += len(target)
    if not observations:
        raise ValueError('No evaluation pairs: at least one trial must be longer than context')
    return {'mse': float(squared_error / observations),
            'copy_last_mse': float(copy_error / observations), 'n_pairs': pairs}


def select_ridge(train_trials, validation_trials, lambdas, context=1):
    """Fit on training only; return the model with the lowest validation MSE.

    Test trials are deliberately not an input. The winner is not refitted on
    validation data. Candidate order breaks ties deterministically.
    """
    lambdas = list(lambdas)
    if not lambdas or any(not np.isfinite(lam) or lam <= 0 for lam in lambdas):
        raise ValueError('Provide at least one finite, positive ridge penalty')
    # Materialize trial lists so generators can be used repeatedly.
    train_trials, validation_trials = list(train_trials), list(validation_trials)
    best_model, best_mse, scores = None, float('inf'), []
    for lam in lambdas:
        model = VAR1(lam=lam).fit(train_trials)
        metrics = evaluate_one_step(model, validation_trials, context)
        scores.append({'lam': float(lam), 'val_mse': metrics['mse'],
                       'n_validation_pairs': metrics['n_pairs']})
        if metrics['mse'] < best_mse:
            best_model, best_mse = model, metrics['mse']
    return best_model, scores
