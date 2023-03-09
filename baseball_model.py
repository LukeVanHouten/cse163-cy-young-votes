import numpy as np
import pandas as pd
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
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
        xgb_r = xgb.XGBRegressor(n_estimators=1000, max_depth=10, eta=0.1,
                                 subsample=1.0, colsample_bytree=1.0)
        xgb_r.fit(self._train_features, self._train_labels)
        vote_pred = xgb_r.predict(self._test_features)
        error = mean_absolute_error(self._test_labels, vote_pred)
        return error

    def random_forest_model(self):
        regressor = RandomForestRegressor(n_estimators=100, random_state=0)
        regressor.fit(self._train_features, self._train_labels)
        vote_pred = regressor.predict(self._test_features)
        error = mean_absolute_error(self._test_labels, vote_pred)
        return error

    def decision_tree_model(self):
        model = DecisionTreeRegressor()
        model.fit(self._train_features, self._train_labels)
        vote_pred = model.predict(self._test_features)
        error = mean_absolute_error(self._test_labels, vote_pred)
        return error

    def compare_eras(self):

        old_data = self._all_cy_young_data[self._all_cy_young_data['Season'] > 2010]
        new_data = self._all_cy_young_data[self._all_cy_young_data['Season'] >= 1990]
        regressor = RandomForestRegressor(n_estimators=100, random_state=0)
        old_data_process = old_data.\
            loc[:, ~old_data.
                columns.isin(["Name", "Season", "L", "G", "GS",
                              "standardized_points"])]
        old_labels = old_data["standardized_points"]
        new_data_process = old_data.\
            loc[:, ~new_data.
                columns.isin(["Name", "Season", "L", "G", "GS",
                              "standardized_points"])]
        new_labels = new_data["standardized_points"]
        old_train_features, old_test_features, old_train_labels, \
            old_test_labels = train_test_split(old_data_process, old_labels,
                                                 test_size=0.2, shuffle=True)
        regressor.fit(old_train_features, old_train_labels)
        vote_pred = regressor.predict(old_test_features)
        error = mean_absolute_error(old_test_labels, vote_pred)
        perm_importance  = permutation_importance(regressor, old_train_features, old_train_labels)
        forest_importances = pd.Series(perm_importance.importances_mean, index=old_data.columns)

        #sorted_indices = np.argsort(importances)[::-1]
        plt.title('Feature Importance')
        fig, ax = plt.subplots()

        forest_importances.plot.bar(yerr=perm_importance.importances_std, ax=ax)
        plt.tight_layout()
        plt.show()


    def starters_vs_relief_pitchers(self):
        pass


PitchingModel().xgboost_model()
