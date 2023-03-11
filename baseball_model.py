import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from baseball_data import PitchingData


class PitchingModel:
    '''
    .
    '''
    def __init__(self, start_year: int, end_year: int) -> None:
        '''
        .
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
        xgb_r = xgb.XGBRegressor(n_estimators=1000, max_depth=10, eta=0.1,
                                 subsample=1.0, colsample_bytree=1.0)
        xgb_r.fit(self._train_features, self._train_labels)
        vote_pred = xgb_r.predict(self._test_features)
        error = mean_absolute_error(self._test_labels, vote_pred)
        return error

    def random_forest_model(self) -> float:
        regressor = RandomForestRegressor(n_estimators=100, random_state=0)
        regressor.fit(self._train_features, self._train_labels)
        vote_pred = regressor.predict(self._test_features)
        error = mean_absolute_error(self._test_labels, vote_pred)
        perm_importance = permutation_importance(regressor,
                                                 self._train_features,
                                                 self._train_labels)
        forest_importances = pd.Series(perm_importance.importances_mean,
                                       index=self._test_features.
                                       columns).sort_values(ascending=False)
        plt.title("Random Forest Feature Importance")
        forest_importances.plot.bar(yerr=perm_importance.importances_std)
        plt.tight_layout()
        plt.show()
        return error

    def decision_tree_model(self) -> float:
        model = DecisionTreeRegressor()
        model.fit(self._train_features, self._train_labels)
        vote_pred = model.predict(self._test_features)
        error = mean_absolute_error(self._test_labels, vote_pred)
        return error

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
