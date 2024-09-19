import math
from expected_exposure import util


#
# metric class to hold raw metric value and supporting data
#
class Metric:
    def __init__(self, name, defaultValue):
        self.name = name
        self.lowerBound = None
        self.upperBound = None
        self.defaultValue = defaultValue
        self.value = 0

    def float(self, normalized=False):
        if normalized:
            if (
                (self.lowerBound == None)
                or (self.upperBound == None)
                or (self.lowerBound == self.upperBound)
            ):
                return self.defaultValue
            else:
                # this will be executed
                # min-max normalization
                return (self.value - self.lowerBound) / (
                    self.upperBound - self.lowerBound
                )
        else:
            return self.value

    def string(self, normalized=False):
        v = self.float(normalized)
        return "%f" % v


#
# INDIVIDUAL METRICS
#


#
# relevance
#
class Relevance(Metric):
    def __init__(self, target, umType, p, u, relevanceLevels, n, m, k, true_n):
        super().__init__("relevance", 1.0)
        self.target = target
        #
        # upper bound: the run reproduces the target exposure
        #
        self.upperBound = util.l2(target, False)
        #
        # lower bound: static ranking in reverse relevance order or 0 if retrieval
        #
        self.lowerBound = 0.0
        if n != math.inf:
            pp = p if (umType == "rbp") else p * u
            dist = []
            for d, e in target.items():
                dist.append(e)
            dist.sort()
            for i in range(len(dist)):
                self.lowerBound = self.lowerBound + pow(pp, i) * dist[i]
        if len(relevanceLevels) <= 1:
            self.upperBound = 0.0
            self.lowerBound = 0.0
        if k is not None:
            if m <= k:
                # if n (inf) rather than true_n is passed, ((k - m) ** 2 / (true_n - m)) will go zero
                # if true_n is passes, it will be a negligible small number but can be more exact
                self.upperBound = m + ((k - m) ** 2 / (true_n - m))
            else:
                self.upperBound = k**2 / m
            self.lowerBound = 0.0

    def compute(self, run):
        self.value = util.dot(self.target, run)


#
# disparity
#
class Disparity(Metric):
    def __init__(self, target, umType, p, u, relevanceLevels, n, m, k, true_n):
        super().__init__("disparity", 0.0)
        #
        # upper bound: static ranking
        #
        if p == 1:
            # k = \sum_i^k (1^2)
            self.upperBound = k
        else:
            self.upperBound = util.geometricSeries(p * p, n)
        #
        # lower bound: uniform random
        # each doc will have exposure of 1/n, which is smaller than 1/(n-m), which is effectively 0 when corpus is large
        #
        self.lowerBound = 0.0
        if n != math.inf:
            pp = p if (umType == "rbp") else p * u
            self.lowerBound = pow(util.geometricSeries(pp, n), 2) / n
            # print(self.lowerBound)

    def compute(self, run):
        self.value = util.l2(run, False)


#
# difference
#
class Difference(Metric):
    def __init__(self, target, umType, p, u, relevanceLevels, n, m, k, true_n):
        super().__init__("difference", 0.0)
        self.target = target
        #
        # lower bound: run exposure reproduces target exposure
        #
        self.lowerBound = 0.0
        #
        # upper bound
        #
        # retrieval setting (n == math.inf)
        #
        # assume that all of the documents in target exposure with values > 0 are at
        # the bottom of the ranking.  the upper bound, then, is decomposed into two
        # parts.  we assume that the exposure at the end of the ranking is effectively
        # zero and the quantity is the exposure "lost" from the relevant documents,
        #
        # \sum_{i=0}^{len(target)} target(i)*target(i)
        #
        # and the second is the exposure "gained" for the nonrelevant documents.  we
        # assume that the corpus is of infinite size and that the relevant documents
        # are all at the end.  we're technically double counting the end but the
        # contribution to the geometric series is so small it should not matter.
        #
        # \sum_{i=0} p^i * p^i
        #
        # reranking setting (n != math.inf)
        #
        # assume the worst exposure is a static ranking in reverse order of relevance
        #
        pp = p if (umType == "rbp") else p * u
        ub = 0.0
        if n == math.inf:
            #
            # retrieval condition
            #
            if p == 1:
                if m <= k:
                    # if n (inf) rather than true_n is passed, ((k - m) ** 2 / (true_n - m)) will go zero
                    # if true_n is passes, it will be a negligible small number but can be more exact
                    ub = m + ((k - m) ** 2 / (true_n - m)) + k
                else:
                    # when m > k
                    # \sum_{i=1}^{m} (k/m)^2 = k^2 / m
                    # k^2 / m is the first part
                    # k is the second part
                    # so (k^2 / m) + k = k * ((k / m) + 1)
                    ub = k * ((k / m) + 1)
            else:
                # contribution lost from relevant documents
                for d, e in target.items():
                    ub += e * e
                # contribution gained from nonrelevant documents
                ub = ub + util.geometricSeries(pp, n)
        else:
            #
            # reranking condition
            #
            # construct the sorted target exposure
            target_vector = []
            for d, e in target.items():
                target_vector.append(e)
            target_vector.sort()
            for i in range(len(target_vector)):
                diff = pow(pp, i) - target_vector[i]
                ub += diff * diff
        self.upperBound = ub

    def compute(self, run):
        self.value = util.distance(self.target, run, False)
