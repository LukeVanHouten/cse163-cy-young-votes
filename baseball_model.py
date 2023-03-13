import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from baseball_data import PitchingData
from sklearn.model_selection import GridSearchCV


class PitchingModel:
    '''
    This class trains and implements machine learning models in order to
    attempt to predict Cy Young vote totals. This class will use a variety of
    different models to predict the Cy Young voting and calculate the
    importance or the features.
    '''
    def __init__(self, start_year: int, end_year: int) -> None:
        '''
        This method initializes the PitchingModel class using methods from
        the PitchingData class to set objects to be used later. A new
        PitchingData object is created using the given start and end year
        parameters
        '''
        pitching = PitchingData(start_year, end_year)
        self._train_features: pd.DataFrame = pitching.train_features()
        self._train_labels: pd.Series = pitching.train_labels()
        self._test_features: pd.DataFrame = pitching.test_features()
        self._test_labels: pd.Series = pitching.test_labels()
        self._before_1990: pd.DataFrame = pitching.before_1990()
        self._after_1989: pd.DataFrame = pitching.after_1989()
        self._starting_pitchers: pd.DataFrame = pitching.starting_pitchers()
        self._relief_pitchers: pd.DataFrame = pitching.relief_pitchers()

    def xgboost_model(self) -> float:
        '''
        Uses the xgboost machine learning model and database to train and
        optimize the parameters for a model to predict Cy Young voting
        results. Returns the mean absolute error and makes a graph of the
        importance's in the model
        '''
        # number of trees in the random forest
        n_estimators = [100, 500, 1000]
        # maximum number of levels allowed in each decision tree
        max_depth = [3, 5, 6, 10, 15, 20]
        subsample = np.arange(0.5, 1.0, 0.1)  # ratio of the training instances
        # shrinks the feature weights to make the boosting process
        # more conservative
        learning_rate = [0.01, 0.1, 0.2, 0.3]
        parameters = {'max_depth': max_depth,
                      'learning_rate': learning_rate,
                      'subsample': subsample,
                      'n_estimators': n_estimators}
        xgb_r = xgb.XGBRegressor()
        grid_search = GridSearchCV(estimator=xgb_r,
                                   param_grid=parameters,
                                   scoring='neg_mean_squared_error')
        grid_search.fit(self._train_features, self._train_labels)
        best_est = grid_search.best_estimator_
        feat_importances = pd.Series(best_est.feature_importances_,
                                     index=self._train_features.columns)
        feat_importances.nlargest(5).plot(kind='barh')
        plt.title("xgboost Importance")
        plt.show()
        return grid_search.best_score_

    def random_forest_model(self) -> float:
        '''
        Uses the random forest machine learning model and database to train
        and optimize the parameters for a model to predict Cy Young voting
        results. Returns the mean absolute error and makes a graph of the
        importance's in the model
        '''
        # number of trees in the random forest
        n_estimators = [5, 20, 50, 100]
        # maximum number of levels allowed in each decision tree
        max_depth = [int(x) for x in
                     np.linspace(10, 120, num=12)]
        # minimum sample number to split a node
        min_samples_split = [2, 6, 10]
        # minimum sample number that can be stored in a leaf node
        min_samples_leaf = [1, 3, 4]
        bootstrap = [True, False]  # method used to sample data points

        parameters = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'bootstrap': bootstrap}
        regressor = RandomForestRegressor()

        grid_search = GridSearchCV(regressor, param_grid=parameters)
        grid_search.fit(self._train_features, self._train_labels)
        best_est = grid_search.best_estimator_
        feat_importances = pd.Series(best_est.feature_importances_,
                                     index=self._train_features.columns)
        feat_importances.nlargest(5).plot(kind='barh')
        plt.title("Random Forest Importance")
        plt.show()
        return grid_search.best_score_

    def decision_tree_model(self) -> float:
        '''
        Uses the decision tree machine learning model and database to train
        and optimize the parameters for a model to predict Cy Young voting
        results. Returns the mean absolute error and makes a graph of the
        importance's in the model
        '''
        model = DecisionTreeRegressor()
        # Strategy to split node, either 'best' or 'random'
        splitters = ["best", "random"]
        # maximum number of levels allowed in each decision tree
        max_depth = [1, 3, 5, 7, 9, 11, 12]
        # minimum sample number that can be stored in a leaf node
        min_samples_leaf = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # The minimum weighted fraction of the sum total of weights
        # (of all the input samples) required to be at a leaf node
        min_weight_fraction_leaf = [0.1, 0.2, 0.3, 0.4, 0.5]
        parameters = {"splitter": splitters,
                      "max_depth": max_depth,
                      "min_samples_leaf": min_samples_leaf,
                      "min_weight_fraction_leaf": min_weight_fraction_leaf}
        tuning_model = GridSearchCV(model, param_grid=parameters,
                                    scoring='neg_mean_squared_error')
        tuning_model.fit(self._train_features, self._train_labels)
        best_est = tuning_model.best_estimator_
        feat_importances = pd.Series(best_est.feature_importances_,
                                     index=self._train_features.columns)
        feat_importances.nlargest(5).plot(kind='barh')
        plt.title("Decision Tree Importance")
        plt.show()
        return tuning_model.best_score_

    def compare_eras(self) -> None:
        '''
        .
        '''
        _, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 13))
        axes = axes.ravel()

        statistics_before_1990 = []
        for i, j in enumerate(self._before_1990.columns[:-1]):
            regressor_before_1990 = LinearRegression()
            regressor_before_1990.\
                fit(self._before_1990[[j]],
                    self._before_1990[self._train_labels.name])
            predictions_before_1990 = regressor_before_1990.\
                predict(self._before_1990[[j]])

            axes[i].scatter(self._before_1990[j],
                            self._before_1990[self._train_labels.name])
            axes[i].plot(self._before_1990[j], predictions_before_1990,
                         color="red")
            axes[i].set_xlabel(j)
            axes[i].set_title("Vote Points Before 1990 vs. " + j)

            r_before_1990, _ =\
                pearsonr(self._before_1990[j],
                         self._before_1990[self._train_labels.name])
            r2_before_1990 = r_before_1990 ** 2
            slope_before_1990 = regressor_before_1990.coef_[0]
            y_intercept_before_1990 = regressor_before_1990.intercept_
            statistics_before_1990.\
                append({"Pitching Stat": j,
                        "Correlation Coefficient": round(r_before_1990, 2),
                        "R^2": round(r2_before_1990, 2),
                        "Slope": round(slope_before_1990, 2),
                        "Y-Intercept": round(y_intercept_before_1990, 2)})

        statistics_after_1989 = []
        for m, n in enumerate(self._after_1989.columns[:-1]):
            regressor_after_1989 = LinearRegression()
            regressor_after_1989.\
                fit(self._after_1989[[n]],
                    self._after_1989[self._train_labels.name])
            predictions_after_1989 = regressor_after_1989.\
                predict(self._after_1989[[n]])

            axes[m + 10].scatter(self._after_1989[n],
                                 self._after_1989[self._train_labels.name])
            axes[m + 10].plot(self._after_1989[n], predictions_after_1989,
                              color='red')
            axes[m + 10].set_xlabel(n)
            axes[m + 10].set_title("Vote Points After 1989 vs. " + n)

            r_after_1989, _ =\
                pearsonr(self._after_1989[n],
                         self._after_1989[self._train_labels.name])
            r2_after_1989 = r_after_1989 ** 2
            slope_after_1989 = regressor_after_1989.coef_[0]
            y_intercept_after_1989 = regressor_after_1989.intercept_
            statistics_after_1989.\
                append({"Pitching Stat": n,
                        "Correlation Coefficient": round(r_after_1989, 2),
                        "R^2": round(r2_after_1989, 2),
                        "Slope": round(slope_after_1989, 2),
                        "Y-Intercept": round(y_intercept_after_1989, 2)})

        plt.tight_layout()
        # plt.savefig("compare_eras.png", bbox_inches="tight")
        plt.show()

        regression_statistics_differences = [
            {"Pitching Stat": a["Pitching Stat"],
             "Correlation Coefficient Difference": round(
                a["Correlation Coefficient"] - b["Correlation Coefficient"], 2
             ),
             "R^2 Difference": round(a["R^2"] - b["R^2"], 2),
             "Slope Difference": round(abs(a["Slope"]) - abs(b["Slope"]), 2),
             "Y-Intercept Difference": round(
                a["Y-Intercept"] - b["Y-Intercept"], 2
             )}
            for b, a in zip(statistics_before_1990, statistics_after_1989)
            if a["Pitching Stat"] == b["Pitching Stat"]
        ]

        print("Linear regression statistics for pitchers before 1990:")
        for x in statistics_before_1990:
            x["Line"] = str(x["Slope"]) + "X + " + str(x["Y-Intercept"])
            if "+ -" in x["Line"]:
                x["Line"] = x["Line"].replace("+ -", "- ")
            del x["Slope"]
            del x["Y-Intercept"]
            print(str(x)[18:-1].replace("'", '').replace(",", " -", 1))
        print()

        print("Linear regression statistics for pitchers 1990 and after:")
        for y in statistics_after_1989:
            y["Line"] = str(y["Slope"]) + "X + " + str(y["Y-Intercept"])
            if "+ -" in y["Line"]:
                y["Line"] = y["Line"].replace("+ -", "- ")
            del y["Slope"]
            del y["Y-Intercept"]
            print(str(y)[18:-1].replace("'", '').replace(",", " -", 1))
        print()

        print("Differences between linear regression statistics for pitchers\
              before 1990 and after 1989:")
        for z in regression_statistics_differences:
            print(str(z)[18:-1].replace("'", '').replace(",", " -", 1))
        return None

    def starters_vs_relievers(self) -> None:
        '''
        .
        '''
        _, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 13))
        axes = axes.ravel()

        statistics_starters = []
        for i, j in enumerate(self._starting_pitchers.columns[:-1]):
            regressor_starters = LinearRegression()
            regressor_starters.\
                fit(self._starting_pitchers[[j]],
                    self._starting_pitchers[self._train_labels.name])
            predictions_starters = regressor_starters.\
                predict(self._starting_pitchers[[j]])

            axes[i].scatter(self._starting_pitchers[j],
                            self._starting_pitchers[self._train_labels.name])
            axes[i].plot(self._starting_pitchers[j],
                         predictions_starters, color="red")
            axes[i].set_xlabel(j)
            axes[i].set_title("Vote Points for Starters vs. " + j)

            r_starters, _ =\
                pearsonr(self._starting_pitchers[j],
                         self._starting_pitchers[self._train_labels.name])
            r2_starters = r_starters ** 2
            slope_starters = regressor_starters.coef_[0]
            y_intercept_starters = regressor_starters.intercept_
            statistics_starters.\
                append({"Pitching Stat": j,
                        "Correlation Coefficient": round(r_starters, 2),
                        "R^2": round(r2_starters, 2),
                        "Slope": round(slope_starters, 2),
                        "Y-Intercept": round(y_intercept_starters, 2)})

        statistics_relievers = []
        for m, n in enumerate(self._relief_pitchers.columns[:-1]):
            regressor_relievers = LinearRegression()
            regressor_relievers.\
                fit(self._relief_pitchers[[n]],
                    self._relief_pitchers[self._train_labels.name])
            predictions_relievers = regressor_relievers.\
                predict(self._relief_pitchers[[n]])

            axes[m + 10].\
                scatter(self._relief_pitchers[n],
                        self._relief_pitchers[self._train_labels.name])
            axes[m + 10].plot(self._relief_pitchers[n],
                              predictions_relievers, color='red')
            axes[m + 10].set_xlabel(n)
            axes[m + 10].set_title("Vote Points for Relievers vs. " + n)

            r_relievers, _ =\
                pearsonr(self._relief_pitchers[n],
                         self._relief_pitchers[self._train_labels.name])
            r2_relievers = r_relievers ** 2
            slope_relievers = regressor_relievers.coef_[0]
            y_intercept_relievers = regressor_relievers.intercept_
            statistics_relievers.\
                append({"Pitching Stat": n,
                        "Correlation Coefficient": round(r_relievers, 2),
                        "R^2": round(r2_relievers, 2),
                        "Slope": round(slope_relievers, 2),
                        "Y-Intercept": round(y_intercept_relievers, 2)})

        plt.tight_layout()
        plt.savefig("compare_eras.png", bbox_inches="tight")
        # plt.show()

        starters_vs_relievers_statistics_differences = [
            {"Pitching Stat": a["Pitching Stat"],
             "Correlation Coefficient Difference": round(
                a["Correlation Coefficient"] - b["Correlation Coefficient"], 2
             ),
             "R^2 Difference": round(a["R^2"] - b["R^2"], 2),
             "Slope Difference": round(abs(a["Slope"]) - abs(b["Slope"]), 2),
             "Y-Intercept Difference": round(
                a["Y-Intercept"] - b["Y-Intercept"], 2
             )}
            for b, a in zip(statistics_starters, statistics_relievers)
            if a["Pitching Stat"] == b["Pitching Stat"]
        ]

        print("Linear regression statistics for starting pitchers:")
        for x in statistics_starters:
            x["Line"] = str(x["Slope"]) + "X + " + str(x["Y-Intercept"])
            if "+ -" in x["Line"]:
                x["Line"] = x["Line"].replace("+ -", "- ")
            del x["Slope"]
            del x["Y-Intercept"]
            print(str(x)[18:-1].replace("'", '').replace(",", " -", 1))
        print()

        print("Linear regression statistics for relief pitchers:")
        for y in statistics_relievers:
            y["Line"] = str(y["Slope"]) + "X + " + str(y["Y-Intercept"])
            if "+ -" in y["Line"]:
                y["Line"] = y["Line"].replace("+ -", "- ")
            del y["Slope"]
            del y["Y-Intercept"]
            print(str(y)[18:-1].replace("'", '').replace(",", " -", 1))
        print()

        print("Differences between linear regression statistics for starters\
 and relievers:")
        for z in starters_vs_relievers_statistics_differences:
            print(str(z)[18:-1].replace("'", '').replace(",", " -", 1))
        return None
