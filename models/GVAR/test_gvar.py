# python -m unittest test_gvar -v   (from models/GVAR)
import unittest

import numpy as np
import torch

from gvar import GVAR
from gvar_nonlinear import NonlinearGVAR
from gvar_direct import DirectGVAR


def random_shift(N, E, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, N, (2, E), generator=g)
    idx = idx[:, idx[0] != idx[1]]
    return torch.sparse_coo_tensor(idx, torch.rand(idx.shape[1], generator=g), (N, N)).coalesce()


class TestGVAR(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.N, self.E, self.P, self.K, self.Q = 6, 3, 3, 2, 4
        self.S = random_shift(self.N, 12, 0)
        self.xs = [torch.randn(L, self.N) for L in (20, 15, 9)]
        self.us = [torch.randn(len(x), self.E) for x in self.xs]

    # all (t, n) targets in one design matrix, solved with lstsq
    def brute_force(self, model, lam, lam_exo):
        N, t0 = self.N, max(self.P, self.Q - 1)
        rows, ys = [], []
        for x, u in zip(self.xs, self.us):
            for t in range(t0, len(x)):
                Z = model.features(x[None, t - self.P:t])[0].reshape(N, -1)  # [N, D]
                ulag = torch.cat([u[t - q] for q in range(self.Q)] + [torch.ones(1)])  # [E1]
                exo = torch.zeros(N, N, len(ulag))
                exo[range(N), range(N)] = ulag  # neuron n only sees its own b_n
                rows.append(torch.cat([Z, exo.reshape(N, -1)], 1))
                ys.append(x[t])
        X, y = torch.cat(rows).double(), torch.cat(ys).double()
        n, D, E1 = len(y), model.taps.numel(), self.Q * self.E + 1
        pen = torch.cat([torch.full((D,), lam), torch.tensor(([lam_exo / N] * (E1 - 1) + [0.]) * N)]).double()
        A = torch.cat([X / n ** 0.5, torch.diag(pen.sqrt())])
        theta = torch.linalg.lstsq(A, torch.cat([y / n ** 0.5, torch.zeros(len(pen), dtype=torch.float64)])).solution
        Bc = theta[D:].reshape(N, E1)
        return theta[:D], Bc[:, :-1], Bc[:, -1]

    def test_closed_form_matches_brute_force(self):
        for leaky in (False, True):
            with self.subTest(leaky=leaky):
                model = GVAR(self.S, self.P, self.K, leaky, self.E, self.Q)
                model.fit(self.xs, lam=0.01, lam_exo=0.3, exo_trials=self.us)
                taps, B, c = self.brute_force(model, 0.01, 0.3)
                torch.testing.assert_close(model.taps.reshape(-1).double(), taps, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(model.B.double(), B, atol=1e-5, rtol=1e-4)
                torch.testing.assert_close(model.c.double(), c, atol=1e-5, rtol=1e-4)

    def test_intercept_only_matches_brute_force(self):
        model = GVAR(self.S, self.P, self.K, True, intercept=True).fit(self.xs, lam=0.01)
        F, y, nid = [], [], []
        for x in self.xs:
            for t in range(self.P, len(x)):
                F.append(model.features(x[None, t - self.P:t])[0].reshape(self.N, -1))
                y.append(x[t])
        F, y = torch.cat(F).double(), torch.cat(y).double()
        X = torch.cat([F, torch.eye(self.N, dtype=torch.float64).repeat(len(y) // self.N, 1)], 1)
        D, n = F.shape[1], len(y)
        A = torch.cat([X / n ** 0.5, torch.cat([0.1 * torch.eye(D), torch.zeros(D, self.N)], 1).double()])
        theta = torch.linalg.lstsq(A, torch.cat([y / n ** 0.5, torch.zeros(D, dtype=torch.float64)])).solution
        torch.testing.assert_close(model.taps.reshape(-1).double(), theta[:D], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(model.c.double(), theta[D:], atol=1e-5, rtol=1e-4)

    def test_without_exogenous_matches_original_normal_equations(self):
        model = GVAR(self.S, self.P, self.K).fit(self.xs, lam=1e-3)
        F, y = [], []
        for x in self.xs:
            for t in range(self.P, len(x)):
                F.append(model.features(x[None, t - self.P:t])[0].reshape(self.N, -1))
                y.append(x[t])
        F, y = torch.cat(F).double(), torch.cat(y).double()
        h = torch.linalg.solve(F.T @ F / len(y) + 1e-3 * torch.eye(F.shape[1], dtype=torch.float64), F.T @ y / len(y))
        torch.testing.assert_close(model.h.reshape(-1).double(), h, atol=1e-5, rtol=1e-4)

    def test_rollout_uses_exogenous_rows_of_each_step(self):
        model = GVAR(self.S, self.P, self.K, True, self.E, self.Q).fit(self.xs, 0.01, 0.3, self.us)
        x, u, t, H = self.xs[0], self.us[0], 8, 3
        pred = model.rollout(x[None, t - self.P:t], H, u[None, t - self.Q + 1:t + H])[0]
        past = x[t - self.P:t]
        for j in range(H):  # step by step by hand
            step = model(past[None], u[None, t + j - self.Q + 1:t + j + 1])[0]
            torch.testing.assert_close(pred[j], step)
            past = torch.cat([past[1:], step[None]])

    def test_rollout_clamps_at_floor_before_feeding_back(self):
        model = GVAR(self.S, self.P, self.K, True).fit(self.xs)
        model.floor = torch.full((self.N,), -0.1)
        x, H = self.xs[0][None, :self.P], 4
        pred = model.rollout(x, H)
        self.assertTrue((pred >= -0.1).all())
        past = x[0]
        for j in range(H):  # by hand: clamp, then feed back
            step = torch.maximum(model(past[None])[0], model.floor)
            torch.testing.assert_close(pred[0, j], step)
            past = torch.cat([past[1:], step[None]])

    # direct head j: brute-force regression of x_{t+j} on the context, behaviour (column 1) held at the origin
    def test_direct_heads_match_brute_force(self):
        known = np.array([True, False, True])
        model = DirectGVAR(self.S, self.P, self.K, 3, True, self.E, self.Q, known_ahead=known)
        model.solve(model.statistics(self.xs, self.us), lam=0.01, lam_exo=0.3)
        N, E1 = self.N, self.Q * self.E + 1
        for j, head in enumerate(model.heads):
            rows, ys = [], []
            for x, u in zip(self.xs, self.us):
                for t in range(max(self.P + j, self.Q - 1), len(x)):
                    Z = head.features(x[None, t - j - self.P:t - j])[0].reshape(N, -1)
                    lags = [torch.where(torch.from_numpy(known) | (q >= j), u[t - q], u[t - j]) for q in range(self.Q)]
                    ulag = torch.cat(lags + [torch.ones(1)])
                    exo = torch.zeros(N, N, E1)
                    exo[range(N), range(N)] = ulag
                    rows.append(torch.cat([Z, exo.reshape(N, -1)], 1))
                    ys.append(x[t])
            X, y = torch.cat(rows).double(), torch.cat(ys).double()
            n, D = len(y), head.taps.numel()
            pen = torch.cat([torch.full((D,), 0.01), torch.tensor(([0.3 / N] * (E1 - 1) + [0.]) * N)]).double()
            A = torch.cat([X / n ** 0.5, torch.diag(pen.sqrt())])
            theta = torch.linalg.lstsq(A, torch.cat([y / n ** 0.5, torch.zeros(len(pen), dtype=torch.float64)])).solution
            torch.testing.assert_close(head.taps.reshape(-1).double(), theta[:D], atol=1e-5, rtol=1e-4)
            torch.testing.assert_close(head.B.double(), theta[D:].reshape(N, E1)[:, :-1], atol=1e-5, rtol=1e-4)

    # direct rollout = each head on the same context, nothing fed back
    def test_direct_rollout_uses_only_the_context(self):
        model = DirectGVAR(self.S, self.P, self.K, 3, False, self.E, self.Q)
        model.solve(model.statistics(self.xs, self.us))
        x, u, t = self.xs[0], self.us[0], 9
        pred = model.rollout(x[None, t - self.P:t], 3, u[None, t - self.Q + 1:t + 3])[0]
        for j in range(3):
            torch.testing.assert_close(pred[j], model.heads[j](x[None, t - self.P:t], u[None, t + j - self.Q + 1:t + j + 1])[0])

    # zero-initialised readout: the untrained network equals its G-VAR
    def test_untrained_nonlinear_equals_gvar(self):
        for gated in (False, True):
            with self.subTest(gated=gated):
                linear = GVAR(self.S, self.P, self.K, True, self.E, self.Q).fit(self.xs, 0.01, 0.3, self.us)
                model = NonlinearGVAR(linear, hidden=4, layers=2, gated=gated, exo_dim=3)
                x, u, t = self.xs[0], self.us[0], 10
                args = (x[None, t - self.P:t], u[None, t - self.Q + 1:t + 1])
                torch.testing.assert_close(model(*args), linear(*args))
                model(*args).sum().backward()  # gradients reach the network, not the G-VAR
                self.assertIsNotNone(model.readout.W.grad)
                self.assertIsNone(linear.h.grad)
                self.assertIsNotNone(model.exo_proj.weight.grad)

    # rollout loss: pure rollout equals the evaluation rollout's error; full teacher forcing equals one-step errors
    def test_rollout_loss_matches_rollout_and_teacher_forcing(self):
        from gvar_nonlinear import rollout_loss
        linear = GVAR(self.S, self.P, self.K, False, self.E, self.Q).fit(self.xs, 0.01, 0.3, self.us)
        model = NonlinearGVAR(linear, hidden=4, layers=1, exo_dim=3)
        torch.nn.init.normal_(model.readout.W)
        x, u, t, H = self.xs[0], self.us[0], 9, 3
        xp, uw, y = x[None, t - self.P:t], u[None, t - self.Q + 1:t + H], x[None, t:t + H]
        with torch.no_grad():
            pure = ((model.rollout(xp, H, uw) - y) ** 2).mean()
            forced = torch.stack([((model(x[None, t + j - self.P:t + j], u[None, t + j - self.Q + 1:t + j + 1]) - y[:, j]) ** 2).mean()
                                  for j in range(H)]).mean()
            torch.testing.assert_close(rollout_loss(model, xp, uw, y), pure)
            torch.testing.assert_close(rollout_loss(model, xp, uw, y, p_teacher=1.0), forced)

    # direct network: untrained equals the direct G-VAR, one output per horizon
    def test_untrained_direct_nonlinear_equals_direct_gvar(self):
        linear = DirectGVAR(self.S, self.P, self.K, 3, False, self.E, self.Q)
        linear.solve(linear.statistics(self.xs, self.us))
        model = NonlinearGVAR(linear, hidden=4, layers=2, exo_dim=3)
        x, u, t = self.xs[0], self.us[0], 9
        args = (x[None, t - self.P:t], u[None, t - self.Q + 1:t + 3])
        torch.testing.assert_close(model(*args, 3), linear(*args, 3))
        self.assertEqual(model(*args, 3).shape, (1, 3, self.N))
        torch.testing.assert_close(model.rollout(*args[:1], 3, args[1]), linear.rollout(*args[:1], 3, args[1]))

    # graph filter layer: sum_k S^k H W_k + b, checked per sample
    def test_graph_filter_matches_explicit_powers(self):
        from gvar_nonlinear import GraphFilter
        f = GraphFilter(2, 3, 5)
        H = torch.randn(4, self.N, 3)
        Sd = self.S.to_dense()
        expected = sum(torch.linalg.matrix_power(Sd, k) @ H @ f.W[k] for k in range(3)) + f.b
        torch.testing.assert_close(f(self.S, H), expected)

    # x_t = 0.3 x_{t-1} + 0.5 S relu(x_{t-1}) + noise
    def test_rectified_fit_recovers_rectified_dynamics(self):
        model = GVAR(self.S, 1, 1, leaky=True)
        x = [torch.randn(self.N)]
        for _ in range(4000):
            prev = x[-1]
            x.append(0.3 * prev + 0.5 * (self.S @ torch.relu(prev)) + 0.5 * torch.randn(self.N))
        model.fit([torch.stack(x)], lam=1e-6)
        np.testing.assert_allclose(model.h.detach().numpy().ravel(), [0.3, 0.0], atol=0.05)
        np.testing.assert_allclose(model.g.detach().numpy().ravel(), [0.0, 0.5], atol=0.05)

if __name__ == '__main__':
    unittest.main()
