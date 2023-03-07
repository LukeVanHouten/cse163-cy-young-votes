import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from baseball_data import PitchingData


class PitchingModel:
    def __init__(self) -> None:
        self._all_cy_young_data: pd.DataFrame =\
            PitchingData().view_merged_data()
        self._train_features: pd.DataFrame = PitchingData().train_features()
        self._train_labels: pd.Series = PitchingData().train_labels()
        self._test_features: pd.DataFrame = PitchingData().test_features()
        self._test_labels: pd.Series = PitchingData().test_labels()

    def xgboost_model(self):
        pass

    def random_forest_model(self):
        pass

    def decision_tree_model(self):
        pass

    def compare_eras(self):
        pass

    def starters_vs_relief_pitchers(self):
        pass
