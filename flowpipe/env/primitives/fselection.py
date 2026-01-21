from copy import deepcopy
from itertools import compress

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_selection import (
    RFE,
    GenericUnivariateSelect,
    SelectFdr,
    SelectFpr,
    SelectFwe,
    SelectKBest,
    SelectPercentile,
    VarianceThreshold,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.svm import SVR

from .primitive import Primitive


class VarianceThresholdPrim(Primitive):
    def __init__(self, random_state=0):
        super(VarianceThresholdPrim, self).__init__(name="VarianceThreshold")
        self.id = 1
        self.gid = 26
        self.PCA_LAPACK_Prim = []
        self.type = "feature selection"
        self.description = "Feature selector that removes all low-variance features."
        self.selector = VarianceThreshold()
        self.accept_type = "c_t"
        self.need_y = True

    def can_accept(self, data):
        return self.can_accept_c(data)

    def is_needed(self, data):
        return True

    def transform(self, train_x, test_x, train_y):
        combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)

        try:
            self.selector.fit(combined)
        except ValueError as exc:
            if "No feature in X meets the variance threshold" in str(exc):
                return (
                    train_x.reset_index(drop=True),
                    test_x.reset_index(drop=True),
                )
            raise

        mask = self.selector.get_support(indices=False)
        if not mask.any():
            return (
                train_x.reset_index(drop=True),
                test_x.reset_index(drop=True),
            )

        final_cols = list(compress(combined.columns, mask))
        transformed = pd.DataFrame(
            self.selector.transform(combined),
            columns=final_cols,
        )

        train_data_x = transformed.iloc[: len(train_x)].reset_index(drop=True)
        test_data_x = transformed.iloc[len(train_x):].reset_index(drop=True)
        return train_data_x, test_data_x



