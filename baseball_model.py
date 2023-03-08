import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from baseball_data import PitchingData
# from sklearn.metrics import accuracy_score


class PitchingModel:
    def __init__(self) -> None:
        self._all_cy_young_data: pd.DataFrame =\
            PitchingData().view_merged_data()
        self._train_features: pd.DataFrame = PitchingData().train_features()
        self._train_labels: pd.Series = PitchingData().train_labels()
        self._test_features: pd.DataFrame = PitchingData().test_features()
        self._test_labels: pd.Series = PitchingData().test_labels()

    def xgboost_model(self):
        xgb_r = xgb.XGBRegressor(n_estimators=1000, max_depth=10, eta=0.1,
                                 subsample=1.0, colsample_bytree=1.0)
        xgb_r.fit(self._train_features, self._train_labels)
        vote_pred = xgb_r.predict(self._test_features)
        # return accuracy_score(vote_pred, self._test_labels)

    def random_forest_model(self):
        regressor = RandomForestRegressor(n_estimators=100, random_state=0)
        regressor.fit(self._train_features, self._train_labels)
        vote_pred = regressor.predict(self._test_features)
        # return accuracy_score(vote_pred, self._test_labels)

    def decision_tree_model(self):
        model = DecisionTreeRegressor()
        model.fit(self._train_features, self._train_labels)
        vote_pred = model.predict(self._test_features)
        # return accuracy_score(vote_pred, self._test_labels)

    def compare_eras(self):
        pass

    def starters_vs_relief_pitchers(self):
        pass


PitchingModel().xgboost_model()
