from dataclasses import dataclass
import numpy as np

_EPS = np.finfo(np.float64).eps
_TINY = np.finfo(np.float64).tiny

def _up(x):
    return np.nextafter(x, np.inf)

def _down(x):
    return np.nextafter(x, -np.inf)

def _array(x):
    if hasattr(x, 'detach'):
        x = x.detach().cpu().numpy()
    a = np.asarray(x, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise ValueError('Curvature inputs must be finite')
    return a

def _gamma(n):
    ne = n * _EPS
    if ne >= 0.01:
        raise ValueError('Matrix too large for this roundoff enclosure')
    return float(_up(ne / (1 - ne)))

def _positive_mm_upper(a, b):
    gamma = _gamma(a.shape[1])
    return _up(_up(a @ b) / (1 - gamma) + a.shape[1] * _TINY)

def _mm_interval(a, ae, b, be):
    center = a @ b
    absolute = _positive_mm_upper(np.abs(a), np.abs(b))
    error = _up(_gamma(a.shape[1]) * absolute + a.shape[1] * _TINY)
    if np.any(ae):
        error = _up(error + _positive_mm_upper(ae, np.abs(b)))
    if np.any(be):
        error = _up(error + _positive_mm_upper(np.abs(a), be))
    if np.any(ae) and np.any(be):
        error = _up(error + _positive_mm_upper(ae, be))
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(error)):
        raise FloatingPointError('Curvature enclosure overflow')
    return (center, error)

def _sum_upper(a, axis=None):
    a = np.asarray(a)
    n = a.size if axis is None else a.shape[axis]
    return _up(np.sum(a, axis=axis) / (1 - _gamma(n)) + n * _TINY)

def _frobenius_upper(a):
    sq = _up(np.abs(a) * np.abs(a))
    return float(_up(np.sqrt(_sum_upper(sq))))

def _gershgorin(mid, radius):
    off = _up(np.abs(mid) + radius)
    np.fill_diagonal(off, 0.0)
    sums = _sum_upper(off, axis=1)
    lo = _down(_down(np.diag(mid) - np.diag(radius)) - sums)
    hi = _up(_up(np.diag(mid) + np.diag(radius)) + sums)
    return (float(np.min(lo)), float(np.max(hi)))

@dataclass(frozen=True)
class SpectralBounds:
    lower: float
    upper: float
    method: str

def symmetric_bounds(a, error=None):
    a = _array(a)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError('Expected a square symmetric matrix')
    e = np.zeros_like(a) if error is None else _array(error)
    direct_lo, direct_hi = _gershgorin(a, e)
    try:
        _, v = np.linalg.eigh((a + a.T) * 0.5)
        zero = np.zeros_like(v)
        av, ave = _mm_interval(a, e, v, zero)
        b, be = _mm_interval(v.T, zero.T, av, ave)
        c, ce = _mm_interval(v.T, zero.T, v, zero)
        blo, bhi = _gershgorin(b, be)
        clo, chi = _gershgorin(c, ce)
        if clo <= 0 or not np.isfinite(chi):
            return SpectralBounds(direct_lo, direct_hi, 'gershgorin_fallback')
        lo = float(_down(blo / (clo if blo <= 0 else chi)))
        hi = float(_up(bhi / (clo if bhi >= 0 else chi)))
        return SpectralBounds(max(lo, direct_lo), min(hi, direct_hi), 'congruence_gershgorin')
    except np.linalg.LinAlgError:
        return SpectralBounds(direct_lo, direct_hi, 'gershgorin_fallback')

def spectral_norm_upper(w):
    w = _array(w)
    if w.ndim != 2:
        raise ValueError('Expected a matrix')
    gram_left = w if w.shape[0] <= w.shape[1] else w.T
    z = np.zeros_like(gram_left)
    gram, ge = _mm_interval(gram_left, z, gram_left.T, z.T)
    gram_bound = symmetric_bounds(gram, ge)
    congruence = float(_up(np.sqrt(max(0.0, gram_bound.upper))))
    row = float(np.max(_sum_upper(np.abs(w), axis=1)))
    col = float(np.max(_sum_upper(np.abs(w), axis=0)))
    one_inf = float(_up(np.sqrt(_up(row * col))))
    return min(congruence, one_inf, _frobenius_upper(w))

def _mul_up(a, b):
    return _up(np.asarray(a) * np.asarray(b))

def _dot_up(a, b):
    return float(_sum_upper(_mul_up(a, b)))

class CurvatureBounds:
    """Cache global alpha for fused sigmoid/tanh/sine NNs and degree <= 2 CCPs."""

    def __init__(self, net):
        self.cache = {}
        self.is_nn = all(hasattr(net, k) for k in ("l1", "lout", "n_hidden_layers"))
        if self.is_nn:
            if getattr(net, "BN", False):
                raise ValueError("Fuse batch normalization before constructing curvature bounds")
            constants = {"sigmoid": (0.25, 0.09622504486493766),
                         "tanh": (1.0, 1.5396007178390026), "sin": (1.0, 1.0)}
            if net.act_name not in constants:
                raise ValueError("Supported activations: sigmoid, tanh, sin")
            self.max_grad, self.max_hess = constants[net.act_name]
            self.weights = [_array(net.l1.weight)] + [
                _array(getattr(net, "lhidden%d" % i).weight)
                for i in range(1, net.n_hidden_layers + 1)]
            self.norms = [spectral_norm_upper(w) for w in self.weights]
            self.output = _array(net.lout.weight)
        elif hasattr(net, "U1") and int(net.n_degree) in (1, 2):
            self.degree = int(net.n_degree)
            self.output = _array(net.C.weight)
            if self.degree == 2:
                u1 = _array(net.U1.weight)
                u2 = _array(net.U1.weight if net.weight_sharing else net.U2.weight)
                self.f = np.concatenate([u1, u2], axis=0).T
                self.q, self.r = np.linalg.qr(self.f, mode="reduced")
                qr, error = _mm_interval(self.q, np.zeros_like(self.q),
                                         self.r, np.zeros_like(self.r))
                residual = np.maximum(np.abs(_down(self.f - _up(qr + error))),
                                      np.abs(_up(self.f - _down(qr - error))))
                self.residual_norm = _frobenius_upper(residual)
                self.q_norm = spectral_norm_upper(self.q)
                self.r_norm = spectral_norm_upper(self.r)
        else:
            raise ValueError("Expected a smooth NeuralNet or degree <= 2 CCP")

    def alpha(self, target, rival):
        key = int(target), int(rival)
        if key not in self.cache:
            self.cache[key] = self._nn(*key) if self.is_nn else self._quadratic(*key)
        return self.cache[key]

    def _nn(self, target, rival):
        r = [self.norms[0]]
        for norm in self.norms[1:]:
            r.append(float(_mul_up(_mul_up(r[-1], self.max_grad), norm)))
        c = self.output[target] - self.output[rival]
        s = [np.maximum(np.abs(_down(c)), np.abs(_up(c)))]
        for w in self.weights[:0:-1]:
            s.append(_mul_up(self.max_grad,
                            _positive_mm_upper(s[-1][None, :], np.abs(w))[0]))
        terms = [_mul_up(_mul_up(ri, ri), np.max(si)) for ri, si in zip(r, reversed(s))]
        k = float(_mul_up(_sum_upper(terms), self.max_hess))
        return float(_up(k / 2))

    def _quadratic(self, target, rival):
        if self.degree == 1:
            return 0.0
        c = self.output[target] - self.output[rival]
        ce = np.maximum(np.abs(_up(c) - c), np.abs(c - _down(c)))
        h = len(c)
        j, je = np.zeros((2 * h, 2 * h)), np.zeros((2 * h, 2 * h))
        idx = np.arange(h)
        j[idx, h + idx] = j[h + idx, idx] = c
        je[idx, h + idx] = je[h + idx, idx] = ce
        rj, rje = _mm_interval(self.r, np.zeros_like(self.r), j, je)
        b, be = _mm_interval(rj, rje, self.r.T, np.zeros_like(self.r.T))
        bounds = symmetric_bounds(b, be)
        # F=QR+E: bound QR J (QR)^T and its residual without exact orthogonality.
        lower_main = float(_down(min(0.0, bounds.lower) * _mul_up(self.q_norm, self.q_norm)))
        jnorm = float(np.max(_up(np.abs(c) + ce)))
        qrnorm = float(_mul_up(self.q_norm, self.r_norm))
        error = _mul_up(jnorm, _mul_up(self.residual_norm,
                                     _up(2 * qrnorm + self.residual_norm)))
        lower = float(_down(lower_main - error))
        return float(_up(max(0.0, -lower) / 2))
