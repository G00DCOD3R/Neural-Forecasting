"""Population VAR with independent self-history plus a low-rank component.

For centered lagged predictors Z and targets Y, fit B=E(d)+A V.T, where
E stacks P diagonal matrices and rank(A V.T)<=r. The objective is
||Y-ZB||_F^2/M + lam_self*||d||_F^2 + lam_population*||A V.T||_F^2.
The intercept is recovered from training means and is not penalized.
Alternating exact conditional updates need not find the joint global optimum.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh

try:
    from .independent_ar import IndependentAR
    from .var_p import VAR, VARStatistics
except ImportError:
    from independent_ar import IndependentAR
    from var_p import VAR, VARStatistics


def target_energy(trials, stats):
    """Centered target sum of squares / M for the SAME within-trial fit rows."""
    count, total = 0, 0.0
    for trial in trials:
        y = np.asarray(trial, dtype=np.float64)[stats.start:]
        total += float(np.square(y-stats.target_mean).sum())
        count += len(y)
    if count != stats.count:
        raise ValueError('Target rows must match the sufficient statistics')
    return total/count


class DiagonalLowRankVAR:
    """Linear VAR(P), with rank constrained only on the stacked added component.

    The added component can also have diagonal entries; it is not forced to be
    off-diagonal. The decomposition is regularized, not a causal attribution.
    Counts report stored factor values, including their rotational redundancy.
    """
    def __init__(self, P=2, rank=16, lam_self=1e-4, lam_population=0.1):
        base = VAR(P, lam_self)
        VAR(P, lam_population)
        if (isinstance(rank, (bool, np.bool_)) or not isinstance(rank, (int, np.integer))
                or rank < 0):
            raise ValueError('Rank must be a nonnegative integer')
        self.P, self.rank = base.P, int(rank)
        self.lam_self, self.lam_population = base.lam, float(lam_population)
        self.lag_weights = self.input_factors = self.output_factors = self.intercept = None
        self.n_train_pairs, self.training_start = 0, None
        self.n_iterations, self.stopping_reason = 0, 'not_fitted'
        self.objective_history = []

    def fit(self, trials, start=None, **options):
        trials = list(trials)
        stats = VAR(self.P).statistics(trials, start)
        return self.fit_statistics(stats, target_energy(trials, stats), **options)

    def fit_statistics(self, stats, energy, max_iter=20, tol=1e-6, on_iteration=None):
        """Fit training statistics; callback may stop after any complete iterate.

        Iteration 0 is independent ridge AR with zero population component.
        Callback sees only this fitted model, iteration and training objective.
        Validation stopping/iterate selection is handled by the caller.
        """
        if not isinstance(stats, VARStatistics) or stats.P != self.P:
            raise ValueError('Statistics must match lag order')
        if (isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer))
                or max_iter < 0 or not np.isfinite(tol) or tol < 0
                or not np.isfinite(energy) or energy < 0):
            raise ValueError('Require max_iter >= 0, finite tol >= 0, and finite target energy >= 0')
        N, P = stats.num_neurons, self.P
        if self.rank > N:
            raise ValueError('Rank exceeds output neuron count')
        own = IndependentAR(P, self.lam_self).fit_statistics(stats)
        self.lag_weights = own.lag_weights.copy()
        self.input_factors = np.zeros((P*N, self.rank))
        self.output_factors = np.zeros((N, self.rank))
        self.n_train_pairs, self.training_start = stats.count, stats.start
        self.n_iterations, self.objective_history = 0, []
        self.stopping_reason = 'max_iter'
        index = np.arange(P)[:, None]*N + np.arange(N)[None, :]
        own_gram = stats.covariance[index[:, None, :], index[None, :, :]].transpose(2, 0, 1)
        own_cross = stats.cross_covariance[index, np.arange(N)[None, :]]
        regularized_own = own_gram + self.lam_self*np.eye(P)

        def update_intercept():
            self.intercept = stats.target_mean - (stats.past_mean.reshape(P, N)*self.lag_weights).sum(0)
            self.intercept -= (stats.past_mean@self.input_factors)@self.output_factors.T

        def objective():
            # Trace expansion avoids constructing a session-sized design or
            # a dense population coefficient matrix for objective evaluation.
            A, V, d = self.input_factors, self.output_factors, self.lag_weights
            CA, VV = stats.covariance@A, V.T@V
            own_population = (CA.reshape(P, N, self.rank)*V[None]).sum(-1)
            q = energy + np.einsum('np,npq,nq->', d.T, own_gram, d.T)
            q += 2*np.sum(d*own_population) + np.sum((A.T@CA)*VV)
            q -= 2*(np.sum(d*own_cross)+np.sum(A*(stats.cross_covariance@V)))
            q += self.lam_self*np.square(d).sum()
            q += self.lam_population*np.sum((A.T@A)*VV)
            return float(q/N)

        def record(iteration):
            update_intercept()
            value = objective()
            if not np.isfinite(value):
                raise FloatingPointError('Training objective became nonfinite')
            if self.objective_history:
                previous = self.objective_history[-1]
                if value > previous+1e-10*max(1.0, abs(previous)):
                    raise FloatingPointError('An exact conditional update increased the training objective')
            self.objective_history.append(value)
            self.n_iterations = iteration
            if on_iteration is not None and on_iteration(self, iteration, value) is False:
                self.stopping_reason = 'callback'
                return False
            return True

        if not record(0):
            return self
        if not self.rank:
            self.stopping_reason = 'independent_ar'
            return self
        if not max_iter:
            return self
        regularized = stats.covariance.copy()
        regularized.flat[::P*N+1] += self.lam_population
        factor = cho_factor(regularized, lower=True)
        for iteration in range(1, max_iter+1):
            # Exact reduced-rank ridge update for targets residualized by E(d).
            CE = (stats.covariance.reshape(P*N, P, N)*self.lag_weights[None]).sum(1)
            residual_cross = stats.cross_covariance-CE
            weights = cho_solve(factor, residual_cross)
            output_energy = residual_cross.T@weights
            _, basis = eigh((output_energy+output_energy.T)/2,
                            subset_by_index=[N-self.rank, N-1])
            self.output_factors = basis[:, ::-1].copy()
            self.input_factors = weights@self.output_factors
            # Exact per-neuron ridge update given the population component.
            CA = stats.covariance@self.input_factors
            own_population = (CA.reshape(P, N, self.rank)*self.output_factors[None]).sum(-1)
            rhs = (own_cross-own_population).T
            self.lag_weights = np.linalg.solve(regularized_own, rhs[..., None])[..., 0].T
            if not record(iteration):
                break
            previous, current = self.objective_history[-2:]
            if previous-current <= tol*max(1.0, abs(previous)):
                self.stopping_reason = 'training_objective_tolerance'
                break
        return self

    def _fitted(self):
        if any(value is None for value in (self.lag_weights, self.input_factors,
                                          self.output_factors, self.intercept)):
            raise RuntimeError('Fit before predicting')

    @property
    def coefficients(self):
        self._fitted()
        N = len(self.intercept)
        coefficients = (self.input_factors@self.output_factors.T).reshape(self.P, N, N)
        coefficients[:, np.arange(N), np.arange(N)] += self.lag_weights
        return coefficients

    @property
    def parameter_count(self):
        self._fitted()
        return self.lag_weights.size+self.input_factors.size+self.output_factors.size+self.intercept.size

    def _history(self, history):
        self._fitted()
        x = np.asarray(history, dtype=np.float64)
        if (x.ndim not in (2, 3) or x.shape[-2] < self.P
                or x.shape[-1] != len(self.intercept) or not np.isfinite(x).all()):
            raise ValueError(f'Expected finite chronological history with context >= {self.P}')
        return x[..., -self.P:, :]

    def predict(self, history):
        x = self._history(history)[..., ::-1, :]
        features = x.reshape(-1, self.P*len(self.intercept))
        population = (features@self.input_factors)@self.output_factors.T
        population = population[0] if x.ndim == 2 else population
        return population+(x*self.lag_weights).sum(-2)+self.intercept

    rollout = VAR.rollout
