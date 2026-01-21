from copy import deepcopy

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from .primitive import Primitive


def catch_num(data):
    num_cols = [col for col in data.columns if str(data[col].dtypes) != "object"]
    num_cols.sort()
    cat_cols = [col for col in data.columns if col not in num_cols]
    cat_train_x = data[cat_cols]
    num_train_x = data[num_cols]
    return cat_train_x, num_train_x


class ImputerMean(Primitive):
    def __init__(self, random_state=0):
        super(ImputerMean, self).__init__(name="ImputerMean")
        self.id = 1
        self.gid = 1
        self.hyperparams = []
        self.type = "ImputerNum"
        self.description = (
            "Imputation transformer for completing missing values by mean."
        )
        self.imp = SimpleImputer()
        self.accept_type = "c"
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        # 只在“存在数值列 且 数值列中有缺失”时需要
        num_cols = list(data._get_numeric_data().columns)
        if not num_cols:
            return False
        with pd.option_context("mode.use_inf_as_na", True):
            return data[num_cols].isna().any().any()

    def transform(self, train_x, test_x, train_y):
        cat_trainX, num_trainX = catch_num(train_x)
        cat_testX, num_testX = catch_num(test_x)

        # 无数值列：直接早退，避免 SimpleImputer 报错
        if num_trainX.shape[1] == 0:
            return train_x.reset_index(drop=True), test_x.reset_index(drop=True)

        self.imp.fit(num_trainX)

        # 训练集
        num_trainX = self.imp.fit_transform(num_trainX)
        num_trainX = pd.DataFrame(num_trainX).reset_index(drop=True).infer_objects()
        num_trainX.columns = [f"num_{i}" for i in num_trainX.columns]
        train_data_x = pd.concat(
            [cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)], axis=1
        )

        # 测试集
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        num_testX.columns = [f"num_{i}" for i in num_testX.columns]
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )
        return train_data_x, test_data_x


class ImputerMedian(Primitive):
    def __init__(self, random_state=0):
        super(ImputerMedian, self).__init__(name="ImputerMedian")
        self.id = 2
        self.gid = 2
        self.hyperparams = []
        self.type = "ImputerNum"
        self.description = (
            "Imputation transformer for completing missing values by median."
        )
        self.imp = SimpleImputer(strategy="median")
        self.accept_type = "c"
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        num_cols = list(data._get_numeric_data().columns)
        if not num_cols:
            return False
        with pd.option_context("mode.use_inf_as_na", True):
            return data[num_cols].isna().any().any()

    def transform(self, train_x, test_x, train_y):
        cat_trainX, num_trainX = catch_num(train_x)
        cat_testX, num_testX = catch_num(test_x)

        if num_trainX.shape[1] == 0:
            return train_x.reset_index(drop=True), test_x.reset_index(drop=True)

        self.imp.fit(num_trainX)

        # 训练集
        num_trainX = self.imp.fit_transform(num_trainX)
        num_trainX = pd.DataFrame(num_trainX).reset_index(drop=True).infer_objects()
        num_trainX.columns = [f"num_{i}" for i in num_trainX.columns]
        train_data_x = pd.concat(
            [cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)], axis=1
        )

        # 测试集
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        num_testX.columns = [f"num_{i}" for i in num_testX.columns]
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )
        return train_data_x, test_data_x


class ImputerNumPrim(Primitive):
    def __init__(self, random_state=0):
        super(ImputerNumPrim, self).__init__(name="ImputerNumMode")
        self.id = 4
        self.gid = 4
        self.hyperparams = []
        self.type = "ImputerNum"
        self.description = (
            "Imputation transformer for completing missing values by mode."
        )
        self.imp = SimpleImputer(strategy="most_frequent")
        self.accept_type = "c"
        self.need_y = False

    def can_accept(self, data):
        return True

    def is_needed(self, data):
        num_cols = list(data._get_numeric_data().columns)
        if not num_cols:
            return False
        with pd.option_context("mode.use_inf_as_na", True):
            return data[num_cols].isna().any().any()

    def transform(self, train_x, test_x, train_y):
        cat_trainX, num_trainX = catch_num(train_x)
        cat_testX, num_testX = catch_num(test_x)

        if num_trainX.shape[1] == 0:
            return train_x.reset_index(drop=True), test_x.reset_index(drop=True)

        self.imp.fit(num_trainX)

        # 训练集
        num_trainX = self.imp.fit_transform(num_trainX)
        num_trainX = pd.DataFrame(num_trainX).reset_index(drop=True).infer_objects()
        num_trainX.columns = [f"num_{i}" for i in num_trainX.columns]
        train_data_x = pd.concat(
            [cat_trainX.reset_index(drop=True), num_trainX.reset_index(drop=True)], axis=1
        )

        # 测试集
        num_testX = self.imp.fit_transform(num_testX)
        num_testX = pd.DataFrame(num_testX).reset_index(drop=True).infer_objects()
        num_testX.columns = [f"num_{i}" for i in num_testX.columns]
        test_data_x = pd.concat(
            [cat_testX.reset_index(drop=True), num_testX.reset_index(drop=True)], axis=1
        )
        return train_data_x, test_data_x
