"""Finite-difference check of every autodiff op, composed as in the model."""
import numpy as np
from autodiff import T, tanh, sigmoid, softmax, weighted_sum, concat, cross_entropy, soft_cross_entropy, mean

rng = np.random.default_rng(0)
B, N, D, H = 3, 5, 4, 6
A = T(rng.normal(size=(D + N, H)))
Bp = T(rng.normal(size=(1, H)))
Q = T(rng.normal(size=(H, N)))
X0 = rng.normal(size=(B, N, D))
p0 = rng.normal(size=(B, N))
tgt = rng.integers(0, N, size=B)
tp = rng.dirichlet(np.ones(N), size=B)
params = [A, Bp, Q]


def f():
    X = T(X0)
    h = T(np.zeros((B, H)))
    losses = []
    for _ in range(3):
        a = softmax(h @ Q + T(p0))
        gl = weighted_sum(a, X)
        z = concat([gl, T(p0)]) @ A + Bp
        h = tanh(z) * sigmoid(z) + h * 0.5
        losses.append(cross_entropy(h @ Q, tgt))
        losses.append(soft_cross_entropy(h @ Q, tp) * 0.3)
    return mean(losses)


loss = f()
loss.backprop()
analytic = [p.g.copy() for p in params]
worst = 0.0
for p, ga in zip(params, analytic):
    for idx in np.ndindex(p.v.shape):
        old = p.v[idx]
        p.v[idx] = old + 1e-6
        lp = f().v
        p.v[idx] = old - 1e-6
        lm = f().v
        p.v[idx] = old
        num = (lp - lm) / 2e-6
        rel = abs(num - ga[idx]) / max(1e-8, abs(num) + abs(ga[idx]))
        worst = max(worst, rel)
print(f"worst relative error: {worst:.2e}")
assert worst < 1e-5, "gradient check failed"
print("gradient check passed")
