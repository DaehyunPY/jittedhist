from typing import Tuple, Callable, Any

from numpy import ndarray, linspace, digitize, zeros
from numba import jit

from pyspark.rdd import RDD


__ALL__ = ['histogram1d', 'histogram', 'histogram2d']


@jit(nopython=True, nogil=True)
def com(h1, h2):
    return h1 + h2


def histogram1d(fr: float, to: float, bins: int) -> Callable[[Any], Tuple[ndarray, ndarray]]:
    @jit(['uint64[:](uint64[:], float64[:])'], nopython=True, nogil=True)
    def inc(init, values):
        binned = digitize(values, linspace(fr, to, bins+1))
        for i in binned:
            init[i] += 1
        return init

    def ret(rdd: RDD) -> Tuple[ndarray, ndarray]:
        return rdd.aggregate(zeros(bins+2, dtype='uint64'), inc, com), linspace(fr, to, bins+1)
    return ret
histogram = histogram1d


def histogram2d(xfr: float, xto: float, xbins: int,
                yfr: float, yto: float, ybins: int) -> Callable[[Any], Tuple[ndarray, ndarray, ndarray]]:
    @jit(['uint64[:, :](uint64[:, :], float64[:, :])'], nopython=True, nogil=True)
    def inc(init, values):
        xbinned = digitize(values[0, :], linspace(xfr, xto, xbins+1))
        ybinned = digitize(values[1, :], linspace(yfr, yto, ybins+1))
        for x, y in zip(xbinned, ybinned):
            init[x, y] += 1
        return init

    def ret(rdd: RDD) -> Tuple[ndarray, ndarray, ndarray]:
        return (rdd.aggregate(zeros((xbins+2, ybins+2), dtype='uint64'), inc, com),
                linspace(xfr, xto, xbins+1), linspace(yfr, yto, ybins+1))
    return ret
