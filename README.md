# jittedhist

Example

```python
from jittedhist import histogram2d
from numpy import ndarray, array
from pyspark.sql import Row
from pyspark.rdd import RDD
import matplotlib.pyplot as plt


def intersting_data(r: Row) -> ndarray:
    return array((r['interest0'], r['interest1']))


rdd: RDD
hist, xedges, yedges = histogram2d(
    rdd.map(interesting_data),
    xfr=0, xto=8000, xbins=1000,
    yfr=-50, yto=50, ybins=200)

print(hist.shape)  # xbins+2, ybins+2
plt.figure()
plt.pcolormesh(xedges, yedges, hist[1:-1, 1:-1].T)
plt.show()
```

To install `pip install git+https://github.com/DaehyunPY/jittedhist.git`
