import math


#
# function: l2
#
# returns sqrt(sum_i x_i*x_i) unless you request dropping sqrt
#
def l2(x, sqrt=True):
    if len(x) == 0:
        return math.inf
    retval = 0.0
    for k, v in x.items():
        retval = retval + v * v
    if retval <= 0:
        return math.inf
    return math.sqrt(retval) if sqrt else retval


#
# function: distance
#
# compute euclidean distance unless you request dropping the sqrt
#
def distance(x, y, sqrt=True):
    retval = 0.0
    keys = set(x) | set(y)
    for k in keys:
        x_k = x[k] if (k in x) else 0.0
        y_k = y[k] if (k in y) else 0.0
        d = x_k - y_k
        retval = retval + d * d
    return math.sqrt(retval) if sqrt else retval


#
# function: dot
#
# returns sum_i x_i * y_i
#
def dot(x, y):
    retval = 0.0
    keys = set(x) & set(y)
    for k in keys:
        v = x[k] * y[k]
        retval = retval + v
    return retval


#
# function: geometricSeries
#
# return \sum_{i=0}^n p^i
#


def geometricSeries(p, n):
    if n == math.inf:
        return 1.0 / (1.0 - p)
    else:
        if p == 1.0:
            return n
        return (1.0 - pow(p, n)) / (1.0 - p)
