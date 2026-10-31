"""Choose ridge strength using validation trials, then test the winner once.

This is a synthetic workflow check, not a MICrONS experiment.
"""
import argparse

try:
    from .selection import evaluate_one_step, select_ridge
    from .synthetic_example import generate_trials
except ImportError:
    from selection import evaluate_one_step, select_ridge
    from synthetic_example import generate_trials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lambdas', type=float, nargs='+',
                        default=[1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    trials, _, _ = generate_trials(seed=args.seed)
    train, validation, test = trials[:14], trials[14:17], trials[17:]

    model, scores = select_ridge(train, validation, args.lambdas)
    print('SYNTHETIC RIDGE SELECTION ONLY: 3 neurons, one lag')
    print('Train/validation/test trials: 14/3/3')
    print('\nEach candidate is fitted on training trials only:')
    print(f'{"lambda":>12} {"validation MSE":>18}')
    for row in scores:
        print(f'{row["lam"]:12g} {row["val_mse"]:18.9f}')
    print(f'\nSelected lambda: {model.lam:g} (lowest validation MSE)')

    # No test score is computed during candidate selection.
    result = evaluate_one_step(model, test)
    print(f'Training pairs: {model.n_train_pairs}; validation pairs: {scores[0]["n_validation_pairs"]}')
    print(f'Final test pairs: {result["n_pairs"]}')
    print(f'Final held-out VAR MSE:       {result["mse"]:.6f}')
    print(f'Final held-out copy-last MSE: {result["copy_last_mse"]:.6f}')
    print('\nThis demonstrates the selection procedure. The chosen penalty is not a MICrONS recommendation.')


if __name__ == '__main__':
    main()
