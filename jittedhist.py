from functools import singledispatch
from typing import Tuple, Iterable

from dask.bag import Bag
from numba import jit
from numpy import ndarray, linspace, digitize, zeros, uint64, column_stack
from pyspark.rdd import RDD

__all__ = ['histogram1d', 'histogram', 'histogram2d']


@jit(nopython=True, nogil=True)
def com(h1, h2):
    return h1 + h2


@singledispatch
def histogram1d(data, *args, **kwargs):
    raise ValueError("Not supported data type: '{}'".format(data))


@histogram1d.register(RDD)
def _(data: RDD, fr: float, to: float, bins: int) -> Tuple[ndarray, ndarray]:
    @jit(['uint64[:](uint64[:], int64)',
          'uint64[:](uint64[:], float64)'], nopython=True, nogil=True)
    def inc(init: ndarray, value: float) -> ndarray:
        binned = digitize([value], linspace(fr, to, bins + 1))
        for i in binned:
            init[i] += 1
        return init

    hist = data.aggregate(zeros(bins + 2, dtype=uint64), inc, com)
    return hist, linspace(fr, to, bins + 1)


@histogram1d.register(Bag)
def _(data: Bag, fr: float, to: float, bins: int) -> Tuple[ndarray, ndarray]:
    # @jit(nopython=True, nogil=True)  # todo: jit this function
    def inc(values: Iterable[float]) -> ndarray:
        binned = digitize(values, linspace(fr, to, bins + 1))
        init = zeros(bins + 2, dtype=uint64)
        for i in binned:
            init[i] += 1
        return init

    hist = data.reduction(inc, sum)
    return hist, linspace(fr, to, bins + 1)


histogram = histogram1d


@singledispatch
def histogram2d(data, *args, **kwargs):
    raise ValueError("Not supported data type: '{}'".format(data))


@histogram2d.register(RDD)
def _(data: RDD, xfr: float, xto: float, xbins: int,
      yfr: float, yto: float, ybins: int) -> Tuple[ndarray, ndarray, ndarray]:
    @jit(['uint64[:, :](uint64[:, :], int64[:])',
          'uint64[:, :](uint64[:, :], float64[:])'], nopython=True, nogil=True)
    def inc(init: ndarray, value: Tuple[float, float]) -> ndarray:
        xbinned = digitize([value[0]], linspace(xfr, xto, xbins + 1))
        ybinned = digitize([value[1]], linspace(yfr, yto, ybins + 1))
        for x, y in zip(xbinned, ybinned):
            init[x, y] += 1
        return init

    hist = data.aggregate(zeros((xbins + 2, ybins + 2), dtype=uint64), inc, com)
    return hist, linspace(xfr, xto, xbins + 1), linspace(yfr, yto, ybins + 1)


@histogram2d.register(Bag)
def _(data: Bag, xfr: float, xto: float, xbins: int,
      yfr: float, yto: float, ybins: int) -> Tuple[ndarray, ndarray, ndarray]:
    # @jit(nopython=True, nogil=True)  # todo: jit this function
    def inc(values: Iterable[Tuple[float, float]]) -> ndarray:
        xvalues, yvalues = column_stack(values)
        xbinned = digitize(xvalues, linspace(xfr, xto, xbins + 1))
        ybinned = digitize(yvalues, linspace(yfr, yto, ybins + 1))
        init = zeros((xbins + 2, ybins + 2), dtype=uint64)
        for x, y in zip(xbinned, ybinned):
            init[x, y] += 1
        return init

    hist = data.reduction(inc, sum)
    return hist, linspace(xfr, xto, xbins + 1), linspace(yfr, yto, ybins + 1)
