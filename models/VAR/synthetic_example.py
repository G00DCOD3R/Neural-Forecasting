"""Small, reproducible correctness example. These are NOT MICrONS recordings."""
import argparse

import numpy as np

try:
    from .var import VAR1
except ImportError:  # Support running this file directly.
    from var import VAR1


def generate_trials(seed=42, count=20, length=500):
    """Generate three-neuron trials with known, stable linear interactions."""
    rng = np.random.default_rng(seed)
    # Rows are source neurons; columns are target neurons.
    # A drives B (0.60), B drives C (0.50), C drives A (0.10).
    coefficients = np.array([[0.45, 0.60, 0.00],
                             [0.00, 0.20, 0.50],
                             [0.10, 0.00, 0.35]])
    population_mean = np.full(3, 3.0)
    intercept = population_mean - population_mean @ coefficients
    trials = []
    for _ in range(count):
        centered = np.zeros((length + 100, 3))
        for t in range(1, len(centered)):
            centered[t] = centered[t-1] @ coefficients + rng.normal(0, 0.1, 3)
        # Discard the initial settling period before retaining each trial.
        trials.append(centered[100:] + population_mean)
    return trials, coefficients, intercept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lam', type=float, default=1e-4,
                        help='Small fixed penalty for coefficient-recovery demonstration; not MICrONS tuning')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    trials, known_coefficients, known_intercept = generate_trials(args.seed)
    # Fixed example: no hyperparameter search. Train on first 14 trials (70%);
    # reserve the next 3 (15%) and evaluate on the last 3 (15%). Equal trial
    # lengths make this simple split useful for demonstrating temporal separation.
    model = VAR1(lam=args.lam).fit(trials[:14])
    previous = np.vstack([x[:-1] for x in trials[17:]])
    targets = np.vstack([x[1:] for x in trials[17:]])
    prediction = model.predict(previous)
    var_mse = np.mean((prediction - targets)**2)
    copy_mse = np.mean((previous - targets)**2)

    np.set_printoptions(precision=3, suppress=True)
    print('SYNTHETIC EXAMPLE ONLY: 3 neurons, 20 independent trials, one lag')
    print(f'Train/validation/test trials: 14/3/3; lambda: {args.lam:g} (fixed)')
    print('Coefficient orientation: rows = source; columns = target; order = A, B, C')
    print('\nKnown coefficients:\n', known_coefficients)
    print('\nLearned coefficients:\n', model.coefficients)
    print('\nKnown intercept:', known_intercept)
    print('Learned intercept:', model.intercept)
    print(f'\nTraining pairs: {model.n_train_pairs}; fitted parameters: {model.parameter_count}')
    print(f'Held-out one-step VAR MSE:       {var_mse:.6f}')
    print(f'Held-out one-step copy-last MSE: {copy_mse:.6f}')
    print(f'Coefficient mean absolute error: {np.mean(abs(model.coefficients-known_coefficients)):.6f}')
    print('\nThis checks the implementation; it is not evidence of performance on MICrONS.')


if __name__ == '__main__':
    main()
