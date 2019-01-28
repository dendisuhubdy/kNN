def count_different_values(k_v1s, k_v2s):
    """kv1s and kv2s should be dictionaries mapping keys to 
    values.  count_different_values() returns the number of keys in 
    k_v1s and k_v2s that don't have the same value"""
    ks = set(k_v1s.iterkeys()) | set(k_v2s.iterkeys())
    return sum(1 for k in ks if k_v1s.get(k) != k_v2s.get(k))


def sum_square_diffs(x0s, x1s):
    """x1s and x2s should be equal-lengthed sequences of numbers.
    sum_square_differences() returns the sum of the squared differences 
    of x1s and x2s."""
    sum((pow(x1-x2,2) for x1,x2 in zip(x1s,x2s)))

def incr(x_c, x, inc=1):
    """increments the value associated with key x in dictionary x_c
    by inc, or sets it to inc if key x is not in dictionary x_c."""
    x_c[x] = x_c.get(x, 0) + inc

def count_items(xs, x_c=None):
    """returns a dictionary x_c whose keys are the items in xs, and 
    whose values are the number of times each item occurs in xs."""
    if x_c == None:
        x_c = {}
    for x in xs:
        incr(x_c, x)
    return x_c

def second(xy):
    """returns the second element in a sequence"""
    return xy[1]

def most_frequent(xs):
    """returns the most frequent item in xs"""
    x_c = count_items(xs)
    return sorted(x_c.iteritems(), key=second, reverse=True)[0][0]


class kNN_classifier:
    """This is a k-nearest-neighbour classifer."""
    def __init__(self, train_data, k, distf):
        self.train_data = train_data
        self.k = min(k, len(train_data))
        self.distf = distf

    def classify(self, x):
        Ns = sorted(self.train_data, 
                    key=lambda xy: self.distf(xy[0], x))
        return most_frequent((y for x,y in Ns[:self.k]))

    def batch_classify(self, xs):
        return [self.classify(x) for x in xs]

def train(train_data, k=1, distf=count_different_values):
    """Returns a kNN_classifer that contains the data, the number of
    nearest neighbours k and the distance function"""
    return kNN_classifier(train_data, k, distf)
