"""Synthetic validation selection followed by five-step recursive forecasting."""
import argparse

try:
    from .evaluation import evaluate_recursive
    from .selection import select_ridge
    from .synthetic_example import generate_trials
except ImportError:
    from evaluation import evaluate_recursive
    from selection import select_ridge
    from synthetic_example import generate_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--context', type=int, default=15)
    parser.add_argument('--lambdas', type=float, nargs='+', default=[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    trials, _, _ = generate_trials(seed=args.seed)
    train, validation, test = trials[:14], trials[14:17], trials[17:]
    model, grid = select_ridge(train, validation, args.lambdas, context=args.context)
    results = evaluate_recursive(model, test, context=args.context, horizon=args.horizon)

    print('SYNTHETIC RECURSIVE FORECAST ONLY: 3 neurons, one lag')
    print(f'Common evaluation context: {args.context} frames; VAR uses only the latest frame')
    print(f'Selected lambda: {model.lam:g} (one-step validation MSE)')
    print(f'Training pairs: {model.n_train_pairs}; validation pairs: {grid[0]["n_validation_pairs"]}')
    print(f'Test forecast windows: {results["n_windows"]}; horizon: {args.horizon}')
    print('\nEach window forecasts all horizons before seeing any future observations:')
    print(f'{"horizon":>8} {"VAR MSE":>12} {"copy MSE":>12} {"VAR Pearson":>14} {"copy Pearson":>14}')
    for h in range(args.horizon):
        var, copy = results['var'], results['copy_last']
        print(f'{h+1:8d} {var["mse"][h]:12.6f} {copy["mse"][h]:12.6f}'
              f' {var["pearson"][h]:14.4f} {copy["pearson"][h]:14.4f}')
    print('\nAll rows use the same test windows. These are synthetic results, not MICrONS results.')


if __name__ == '__main__':
    main()
