import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from baseball_data import PitchingData


class PitchingModel:
    def __init__(self) -> None:
        self._train_features: pd.DataFrame = PitchingData().train_features()
        self._train_labels: pd.Series = PitchingData().train_labels()
        self._test_features: pd.DataFrame = PitchingData().test_features()
        self._test_labels: pd.Series = PitchingData().test_labels()

    def xgboost_model(self):
        pass

    def random_forest_mode(self):
        pass

    def decision_tree_model(self):
        pass
