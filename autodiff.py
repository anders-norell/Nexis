"""Minimal reverse-mode autodiff on NumPy arrays.

Only the operations the Nexis PoC needs. Every op is checked against
finite differences in test_autodiff.py.
"""
import numpy as np


class T:
    """A tensor node: value, gradient, and how to push gradient to parents."""

    def __init__(self, value, parents=(), backward=None):
        self.v = np.asarray(value, dtype=np.float64)
        self.g = None
        self.parents = parents
        self._backward = backward

    @property
    def shape(self):
        return self.v.shape

    # ---- graph traversal -------------------------------------------------
    def backprop(self):
        order, seen = [], set()

        def visit(n):
            if id(n) in seen:
                return
            seen.add(id(n))
            for p in n.parents:
                visit(p)
            order.append(n)

        visit(self)
        self.g = np.ones_like(self.v)
        for n in reversed(order):
            if n._backward is not None and n.g is not None:
                n._backward(n.g)

    def _acc(self, grad):
        grad = _unbroadcast(grad, self.v.shape)
        self.g = grad if self.g is None else self.g + grad

    # ---- operators -------------------------------------------------------
    def __add__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.v + o.v, (self, o))
        out._backward = lambda g: (self._acc(g), o._acc(g))
        return out

    __radd__ = __add__

    def __sub__(self, o):
        o = o if isinstance(o, T) else T(o)
        return self + (o * -1.0)

    def __mul__(self, o):
        o = o if isinstance(o, T) else T(o)
        out = T(self.v * o.v, (self, o))
        out._backward = lambda g: (self._acc(g * o.v), o._acc(g * self.v))
        return out

    __rmul__ = __mul__

    def __matmul__(self, o):
        out = T(self.v @ o.v, (self, o))

        def bw(g):
            self._acc(g @ np.swapaxes(o.v, -1, -2))
            o._acc(np.swapaxes(self.v, -1, -2) @ g)

        out._backward = bw
        return out


def _unbroadcast(grad, shape):
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def tanh(x):
    y = np.tanh(x.v)
    out = T(y, (x,))
    out._backward = lambda g: x._acc(g * (1 - y * y))
    return out


def sigmoid(x):
    y = 1.0 / (1.0 + np.exp(-x.v))
    out = T(y, (x,))
    out._backward = lambda g: x._acc(g * y * (1 - y))
    return out


def softmax(x):
    """Softmax over the last axis."""
    z = x.v - x.v.max(axis=-1, keepdims=True)
    e = np.exp(z)
    y = e / e.sum(axis=-1, keepdims=True)
    out = T(y, (x,))

    def bw(g):
        x._acc(y * (g - (g * y).sum(axis=-1, keepdims=True)))

    out._backward = bw
    return out


def weighted_sum(a, X):
    """a: (B,N) weights, X: (B,N,D) items -> (B,D)."""
    out = T(np.einsum("bn,bnd->bd", a.v, X.v), (a, X))

    def bw(g):
        a._acc(np.einsum("bd,bnd->bn", g, X.v))
        X._acc(np.einsum("bn,bd->bnd", a.v, g))

    out._backward = bw
    return out


def concat(xs):
    """Concatenate along the last axis."""
    out = T(np.concatenate([x.v for x in xs], axis=-1), tuple(xs))
    sizes = np.cumsum([0] + [x.v.shape[-1] for x in xs])

    def bw(g):
        for x, a, b in zip(xs, sizes[:-1], sizes[1:]):
            x._acc(g[..., a:b])

    out._backward = bw
    return out


def cross_entropy(logits, target_idx):
    """Mean cross-entropy; logits (B,K), target_idx (B,) ints."""
    z = logits.v - logits.v.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    B = logits.v.shape[0]
    loss = -logp[np.arange(B), target_idx].mean()
    out = T(loss, (logits,))

    def bw(g):
        p = np.exp(logp)
        p[np.arange(B), target_idx] -= 1.0
        logits._acc(g * p / B)

    out._backward = bw
    return out


def soft_cross_entropy(logits, target_probs):
    """Mean cross-entropy against a target distribution (no grad to target)."""
    z = logits.v - logits.v.max(axis=-1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=-1, keepdims=True))
    B = logits.v.shape[0]
    tp = target_probs
    loss = -(tp * logp).sum(axis=-1).mean()
    out = T(loss, (logits,))

    def bw(g):
        p = np.exp(logp)
        logits._acc(g * (p * tp.sum(axis=-1, keepdims=True) - tp) / B)

    out._backward = bw
    return out


def mean(xs):
    """Mean of a list of scalar tensors."""
    total = xs[0]
    for x in xs[1:]:
        total = total + x
    return total * (1.0 / len(xs))


class Param(T):
    pass


class Adam:
    def __init__(self, params, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, clip=1.0):
        self.p, self.lr, self.b1, self.b2, self.eps, self.clip = params, lr, b1, b2, eps, clip
        self.m = [np.zeros_like(p.v) for p in params]
        self.s = [np.zeros_like(p.v) for p in params]
        self.t = 0

    def step(self):
        self.t += 1
        grads = [p.g if p.g is not None else np.zeros_like(p.v) for p in self.p]
        norm = np.sqrt(sum((g * g).sum() for g in grads))
        scale = min(1.0, self.clip / (norm + 1e-12))
        for i, (p, g) in enumerate(zip(self.p, grads)):
            g = g * scale
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.s[i] = self.b2 * self.s[i] + (1 - self.b2) * g * g
            mh = self.m[i] / (1 - self.b1 ** self.t)
            sh = self.s[i] / (1 - self.b2 ** self.t)
            p.v -= self.lr * mh / (np.sqrt(sh) + self.eps)
            p.g = None
        return norm
