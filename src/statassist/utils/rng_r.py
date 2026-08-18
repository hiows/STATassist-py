"""R-compatible Mersenne-Twister RNG (set.seed / runif / rnorm / sample / rbinom).

Transcription of R src/main/RNG.c, src/nmath/snorm.c, src/nmath/qnorm.c,
src/nmath/rbinom.c, src/main/random.c, src/main/sort.c (revsort).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Generator

import numpy as np

I2_32M1 = 2.328306437080797e-10  # 1 / (2^32 - 1)
BIG = 134217728  # 2^27

N_MT = 624
M_MT = 397
MATRIX_A = 0x9908B0DF
UPPER_MASK = 0x80000000
LOWER_MASK = 0x7FFFFFFF
TEMPERING_MASK_B = 0x9D2C5680
TEMPERING_MASK_C = 0xEFC60000


def _fixup(x: float) -> float:
    if x <= 0.0:
        return 0.5 * I2_32M1
    if (1.0 - x) <= 0.0:
        return 1.0 - 0.5 * I2_32M1
    return x


def _to_uint32(x: int) -> int:
    return int(x) & 0xFFFFFFFF


def _qnorm5(p: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """AS241 normal quantile (R qnorm5, lower_tail=True, log_p=False)."""
    if not math.isfinite(p) or not math.isfinite(mu) or not math.isfinite(sigma):
        return float("nan")
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    if sigma < 0:
        return float("nan")
    if sigma == 0:
        return mu

    p_ = p
    q = p_ - 0.5

    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        val = (
            q
            * (
                (
                    (
                        (
                            (
                                (
                                    (
                                        r * 2509.0809287301226727 + 33430.575583588128105
                                    )
                                    * r
                                    + 67265.770927008700853
                                )
                                * r
                                + 45921.953931549871457
                            )
                            * r
                            + 13731.693765509461125
                        )
                        * r
                        + 1971.5909503065514427
                    )
                    * r
                    + 133.14166789178437745
                )
                * r
                + 3.387132872796366608
            )
            / (
                (
                    (
                        (
                            (
                                (
                                    (
                                        r * 5226.495278852854561 + 28729.085735721942674
                                    )
                                    * r
                                    + 39307.89580009271061
                                )
                                * r
                                + 21213.794301586595867
                            )
                            * r
                            + 5394.1960214247511077
                        )
                        * r
                        + 687.1870074920579083
                    )
                    * r
                    + 42.313330701600911252
                )
                * r
                + 1.0
            )
        )
    else:
        lp = math.log(p_ if q <= 0 else 1.0 - p_)
        r = math.sqrt(-lp)
        if r <= 5.0:
            r -= 1.6
            val = (
                (
                    (
                        (
                            (
                                (
                                    (
                                        r * 7.7454501427834140764e-4 + 0.0227238449892691845833
                                    )
                                    * r
                                    + 0.24178072517745061177
                                )
                                * r
                                + 1.27045825245236838258
                            )
                            * r
                            + 3.64784832476320460504
                        )
                        * r
                        + 5.7694972214606914055
                    )
                    * r
                    + 4.6303378461565452959
                )
                * r
                + 1.42343711074968357734
            ) / (
                (
                    (
                        (
                            (
                                (
                                    (
                                        r * 1.05075007164441684324e-9 + 5.475938084995344946e-4
                                    )
                                    * r
                                    + 0.0151986665636164571966
                                )
                                * r
                                + 0.14810397642748007459
                            )
                            * r
                            + 0.68976733498510000455
                        )
                        * r
                        + 1.6763848301838038494
                    )
                    * r
                    + 2.05319162663775882187
                )
                * r
                + 1.0
            )
        elif r <= 27.0:
            r -= 5.0
            val = (
                (
                    (
                        (
                            (
                                (
                                    (
                                        r * 2.01033439929228813265e-7 + 2.71155556874348757815e-5
                                    )
                                    * r
                                    + 0.0012426609473880784386
                                )
                                * r
                                + 0.026532189526576123093
                            )
                            * r
                            + 0.29656057182850489123
                        )
                        * r
                        + 1.7848265399172913358
                    )
                    * r
                    + 5.4637849111641143699
                )
                * r
                + 6.6579046435011037772
            ) / (
                (
                    (
                        (
                            (
                                (
                                    (
                                        r * 2.04426310338993978564e-15 + 1.4215117583164458887e-7
                                    )
                                    * r
                                    + 1.8463183175100546818e-5
                                )
                                * r
                                + 7.868691311456132591e-4
                            )
                            * r
                            + 0.0148753612908506148525
                        )
                        * r
                        + 0.13692988092273580531
                    )
                    * r
                    + 0.59983220655588793769
                )
                * r
                + 1.0
            )
        else:
            if r >= 6.4e8:
                val = r * math.sqrt(2.0)
            else:
                s2 = -math.ldexp(lp, 1)
                x2 = s2 - math.log(math.pi * 2.0 * s2)
                if r < 36000.0:
                    x2 = s2 - math.log(math.pi * 2.0 * x2) - 2.0 / (2.0 + x2)
                    if r < 840.0:
                        x2 = s2 - math.log(math.pi * 2.0 * x2) + 2.0 * math.log1p(
                            -(1.0 - 1.0 / (4.0 + x2)) / (2.0 + x2)
                        )
                        if r < 109.0:
                            x2 = s2 - math.log(math.pi * 2.0 * x2) + 2.0 * math.log1p(
                                -(1.0 - (1.0 - 5.0 / (6.0 + x2)) / (4.0 + x2)) / (2.0 + x2)
                            )
                            if r < 55.0:
                                x2 = s2 - math.log(math.pi * 2.0 * x2) + 2.0 * math.log1p(
                                    -(
                                        1.0
                                        - (
                                            1.0
                                            - (1.0 - 5.0 / (6.0 + x2)) / (4.0 + x2)
                                        )
                                        / (4.0 + x2)
                                    )
                                    / (2.0 + x2)
                                )
                val = math.sqrt(x2)
        if q < 0.0:
            val = -val

    return mu + sigma * val


def _revsort(a: list[float], ib: list[int]) -> None:
    """In-place descending heapsort (R revsort, 1-based internally)."""
    n = len(a)
    if n <= 1:
        return
    aa = [0.0] + a[:]
    iib = [0] + ib[:]
    l = (n >> 1) + 1
    ir = n
    while True:
        if l > 1:
            l -= 1
            ra = aa[l]
            ii = iib[l]
        else:
            ra = aa[ir]
            ii = iib[ir]
            aa[ir] = aa[1]
            iib[ir] = iib[1]
            ir -= 1
            if ir == 1:
                aa[1] = ra
                iib[1] = ii
                a[:] = aa[1:]
                ib[:] = iib[1:]
                return
        i = l
        j = l << 1
        while j <= ir:
            if j < ir and aa[j] > aa[j + 1]:
                j += 1
            if ra > aa[j]:
                aa[i] = aa[j]
                iib[i] = iib[j]
                i = j
                j = i << 1
            else:
                j = ir + 1
        aa[i] = ra
        iib[i] = ii


class RRandom:
    """R Mersenne-Twister with Inversion normals and Rejection sampling."""

    def __init__(self, seed: int | float | None = None) -> None:
        self.mt: np.ndarray = np.zeros(N_MT, dtype=np.uint32)
        self.mti: int = N_MT + 1
        self._binom_psave = -1.0
        self._binom_nsave = -1
        self._binom_qn = 0.0
        if seed is not None:
            self.set_seed(seed)

    def set_seed(self, seed: int | float) -> None:
        s = _to_uint32(int(seed))
        for _ in range(50):
            s = _to_uint32(69069 * s + 1)
        i_seed = np.zeros(625, dtype=np.int32)
        for j in range(625):
            s = _to_uint32(69069 * s + 1)
            i_seed[j] = np.int32(s if s < 2**31 else s - 2**32)
        i_seed[0] = 624
        self.mti = int(i_seed[0])
        self.mt = i_seed[1:625].astype(np.uint32)

    def _mt_genrand_raw(self) -> float:
        mag01 = (0, MATRIX_A)
        if self.mti >= N_MT:
            if self.mti == N_MT + 1:
                self._mt_sgenrand(4357)
            for kk in range(N_MT - M_MT):
                y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK)
                self.mt[kk] = self.mt[kk + M_MT] ^ ((y >> 1) ^ mag01[y & 1])
            for kk in range(N_MT - M_MT, N_MT - 1):
                y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK)
                self.mt[kk] = self.mt[kk + (M_MT - N_MT)] ^ ((y >> 1) ^ mag01[y & 1])
            y = (self.mt[N_MT - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK)
            self.mt[N_MT - 1] = self.mt[M_MT - 1] ^ ((y >> 1) ^ mag01[y & 1])
            self.mti = 0
        y = int(self.mt[self.mti])
        self.mti += 1
        y ^= y >> 11
        y ^= (y << 7) & TEMPERING_MASK_B
        y ^= (y << 15) & TEMPERING_MASK_C
        y ^= y >> 18
        return float(y) * 2.3283064365386963e-10

    def _mt_sgenrand(self, seed: int) -> None:
        s = _to_uint32(seed)
        for i in range(N_MT):
            self.mt[i] = (s & 0xFFFF0000)
            s = _to_uint32(69069 * s + 1)
            self.mt[i] |= (s & 0xFFFF0000) >> 16
            s = _to_uint32(69069 * s + 1)
        self.mti = N_MT

    def unif_rand(self) -> float:
        return _fixup(self._mt_genrand_raw())

    def norm_rand(self) -> float:
        u1 = self.unif_rand()
        u1 = int(BIG * u1) + self.unif_rand()
        return _qnorm5(u1 / BIG)

    def runif(self, n: int, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
        return min_val + (max_val - min_val) * np.array([self.unif_rand() for _ in range(n)])

    def rnorm(self, n: int, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
        return mean + sd * np.array([self.norm_rand() for _ in range(n)])

    def _rbits(self, bits: int) -> float:
        v = 0
        for n in range(0, bits + 1, 16):
            v1 = int(math.floor(self.unif_rand() * 65536))
            v = 65536 * v + v1
        mask = (1 << bits) - 1
        return float(v & mask)

    def unif_index(self, dn: int) -> int:
        if dn <= 0:
            return 0
        bits = int(math.ceil(math.log2(dn)))
        while True:
            dv = self._rbits(bits)
            if dn > dv:
                return int(dv)

    def sample_int(self, n: int, k: int) -> np.ndarray:
        """R sample.int without replacement (1-based indices)."""
        if k <= 0:
            return np.array([], dtype=int)
        if k > n:
            raise ValueError("cannot take a sample larger than the population")
        if k == 1:
            return np.array([self.unif_index(n) + 1], dtype=int)
        x = list(range(n))
        out = np.empty(k, dtype=int)
        for i in range(k):
            j = self.unif_index(n - i)
            out[i] = x[j] + 1
            x[j] = x[n - i - 1]
        return out

    def sample_replace(self, n: int, k: int) -> np.ndarray:
        """R sample.int with replacement (1-based indices)."""
        return np.array([self.unif_index(n) + 1 for _ in range(k)], dtype=int)

    def sample_prob_replace(self, probs: np.ndarray, k: int) -> np.ndarray:
        """R sample(prob=, replace=TRUE) via ProbSampleReplace."""
        n = len(probs)
        p = probs.astype(float).copy()
        s = p.sum()
        if s <= 0:
            raise ValueError("non-positive probability sum")
        p /= s
        perm = list(range(1, n + 1))
        p_list = p.tolist()
        _revsort(p_list, perm)
        for i in range(1, n):
            p_list[i] += p_list[i - 1]
        ans = []
        nm1 = n - 1
        for _ in range(k):
            r_u = self.unif_rand()
            j = nm1
            for jj in range(nm1):
                if r_u <= p_list[jj]:
                    j = jj
                    break
            else:
                j = nm1
            ans.append(perm[j])
        return np.array(ans, dtype=int)

    def rbinom(self, n_draws: int, size: int, prob: float) -> np.ndarray:
        out = np.empty(n_draws, dtype=int)
        for i in range(n_draws):
            out[i] = self._rbinom_one(size, prob)
        return out

    def _rbinom_one(self, n: int, pp: float) -> int:
        if n == 0 or pp == 0.0:
            return 0
        if pp == 1.0:
            return n
        p = min(pp, 1.0 - pp)
        q = 1.0 - p
        np_val = n * p
        r = p / q
        g = r * (n + 1)

        if pp != self._binom_psave or n != self._binom_nsave:
            self._binom_psave = pp
            self._binom_nsave = n
            if np_val < 30.0:
                self._binom_qn = (
                    q**n if p > 0.25 else math.exp(n * math.log1p(-p))
                )
            else:
                self._binom_qn = 0.0

        if np_val < 30.0:
            qn = self._binom_qn
            while True:
                ix = 0
                f = qn
                u = self.unif_rand()
                while True:
                    if u < f:
                        return n - ix if pp > 0.5 else ix
                    if ix > 110:
                        break
                    u -= f
                    ix += 1
                    f *= g / ix - r
        raise RuntimeError("rbinom BTPE not implemented for np >= 30")

    def shuffle_int(self, n: int) -> np.ndarray:
        """R sample.int(n) — permutation of 1..n."""
        return self.sample_int(n, n)


_current_rng: ContextVar[RRandom | None] = ContextVar("_current_rng", default=None)


def get_rng() -> RRandom:
    rng = _current_rng.get()
    if rng is None:
        rng = RRandom()
        _current_rng.set(rng)
    return rng


@contextmanager
def sa_r_seed(seed: float | None) -> Generator[RRandom, None, None]:
    """Context manager mirroring R sa_preserve_seed + set.seed."""
    if seed is None:
        yield get_rng()
        return
    prev = _current_rng.get()
    rng = RRandom(seed)
    token = _current_rng.set(rng)
    try:
        yield rng
    finally:
        _current_rng.reset(token)
        if prev is not None:
            _current_rng.set(prev)


# Module-level helpers used by simulators (active RNG from context).
def runif(n: int, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    return get_rng().runif(n, min_val, max_val)


def rnorm(n: int, mean: float = 0.0, sd: float = 1.0) -> np.ndarray:
    return get_rng().rnorm(n, mean, sd)


def sample_int(n: int, k: int) -> np.ndarray:
    return get_rng().sample_int(n, k)


def sample_prob_replace(probs: np.ndarray, k: int) -> np.ndarray:
    return get_rng().sample_prob_replace(probs, k)


def rbinom(n_draws: int, size: int, prob: float) -> np.ndarray:
    return get_rng().rbinom(n_draws, size, prob)
