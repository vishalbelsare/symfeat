from itertools import product, chain, combinations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class Base(BaseEstimator, TransformerMixin):
    pass


class ConstantFeature(Base):
    def transform(self, x):
        return np.ones_like(x[:, 0])

    @property
    def name(self):
        return "1"


class SimpleFeature(Base):
    """Base to create polynomial features.
    """
    def __init__(self, exponent, index=0):
        if exponent == 0:
            raise ValueError
        self.exponent = exponent
        self.index = index

    def transform(self, x):
        return x[:, self.index]**self.exponent

    @property
    def name(self):
        if self.exponent == 1:
            return "x_{}".format(self.index)
        else:
            return "x_{}**{}".format(self.index, self.exponent)


class OperatorFeature(Base):
    def __init__(self, feat_cls, operator, operator_name=None):
        self.feat_cls = feat_cls
        self.operator = operator
        self.operator_name = operator_name or str(operator)

    def transform(self, x):
        return self.operator(self.feat_cls.transform(x))

    @property
    def name(self):
        return "{}({})".format(self.operator_name, self.feat_cls.name)


class ProductFeature(Base):
    def __init__(self, feat_cls1, feat_cls2):
        self.feat_cls1 = feat_cls1
        self.feat_cls2 = feat_cls2

    def transform(self, x):
        return self.feat_cls1.transform(x) * self.feat_cls2.transform(x)

    @property
    def name(self):
        return "{}*{}".format(self.feat_cls1.name, self.feat_cls2.name)

def _allfinite(tpl):
    b, x = tpl
    return np.all(np.isfinite(x))

def _take_finite(x):
    return list(filter(_allfinite, x))


class SymbolicFeatures(Base):
    """Main class.
    """
    def __init__(self, exponents, operators):
        self.exponents = exponents
        self.operators = operators

    def transform(self, x):
        _, n_features = x.shape
        # 0) Get constant feature
        const = [(ConstantFeature(), ConstantFeature().transform(x))]
        # 1) Get all simple features
        simple = (SimpleFeature(e, index=i) for e, i in product(self.exponents, range(n_features)))
        simple = _take_finite((s, s.transform(x)) for s in simple)
        # 2) Get all operator features
        operator = (OperatorFeature(s, op, operator_name=op_name) for (s, _), (op_name, op) in product(simple, self.operators.items()))
        operator = _take_finite((o, o.transform(x)) for o in operator)
        # 3) Get all product features
        combs = chain(product(operator, simple), combinations(simple, 2))
        prod = [ProductFeature(feat1, feat2) for (feat1, _) , (feat2, _) in combs]
        prod = _take_finite((p, p.transform(x)) for p in prod)

        all_ = const + simple + operator + prod
        names, features = zip(*[(c.name, np.array(f)) for c,f in all_])
        features = np.array(list(features)).T
        self.names = list(names)
        return features
