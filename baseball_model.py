'''
Aidan Murphy, Luke VanHouten
CSE 163 section AH
13 March 2023

This program implements the class PitchingModel to load in the data used in
our model to predict Cy Young Award vote points based on the pitching
statistics of top-tier pitchers in Major League Baseball, and then run each of
those models, including XGBoost, Random Forest, and Decision trees. It then
plots and creates statistics for different linear regressions to see the
correlation between the vote points and the different pitching statistics used
in the model between different eras or different types of pitchers.
'''

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from baseball_data import PitchingData


class PitchingModel:
    '''
    This class trains and implements machine learning models in order to
    attempt to predict Cy Young vote totals. This class will use a variety of
    different models to predict the Cy Young voting and calculate the
    importance or the features.
    '''
    def __init__(self, start_year: int, end_year: int,
                 id_filename: str) -> None:
        '''
        This method initializes the PitchingModel class using methods from
        the PitchingData class to set objects to be used later. A new
        PitchingData object is created using the given start and end year
        parameters. Then, each method from that file is initialized in order
        to load in the data.
        '''
        # Creates an object for the PitchingData class from the previous file,
        # using the same inputs as this initializor
        pitching = PitchingData(start_year, end_year, id_filename)
        # Brings in all of the methods from this class
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
        This method uses the xgboost machine learning model and database to
        train and optimize the parameters for a model to predict Cy Young vote
        points. Returns the mean absolute error and makes a graph of the
        importances in the model.
        '''
        # The number of trees in the random forest
        n_estimators = [100, 500, 1000]
        # The maximum number of levels allowed in each decision tree
        max_depth = [2, 4, 6, 8]
        # This is the ratio of the training instances
        subsample = np.arange(0.5, 1.0, 0.1)
        # Shrinks the feature weights to make the boosting process more
        # conservative
        learning_rate = [0.01, 0.1, 0.2, 0.3]
        # Creates a dictionary of these hyperparameters to be used in the
        # model
        parameters = {'max_depth': max_depth,
                      'learning_rate': learning_rate,
                      'subsample': subsample,
                      'n_estimators': n_estimators}
        # Initializes the model
        xgb_r = xgb.XGBRegressor()
        # Loads the hyperparameters into a grid search using the XGBoost model
        # and ensures that it is using MAE for the error
        grid_search = GridSearchCV(estimator=xgb_r,
                                   param_grid=parameters,
                                   scoring='neg_mean_absolute_error')
        # Fits the data to the model and the grid search
        grid_search.fit(self._train_features, self._train_labels)
        # Creates the best estimator of the most effective model using the
        # grid search
        best_est = grid_search.best_estimator_
        # Creates the feature importance from the model and plots it
        feat_importances = pd.Series(best_est.feature_importances_,
                                     index=self._train_features.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        # Gives the plot a name and saves it as a .png file
        plt.title("XGBoost Importance")
        plt.savefig("xgboost.png", bbox_inches="tight")
        # Returns the MAE of the model, while making sure that it is positive,
        # as the model is set to use a negative MAE to singify that it is
        # something to be minimized. However, it is less confusing to keep the
        # error positive
        return -grid_search.best_score_

    def random_forest_model(self) -> float:
        '''
        This method uses the random forest machine learning model and database
        to train and optimize the parameters for a model to predict Cy Young
        vote points. Returns the mean absolute error and makes a graph of the
        importance's in the model.
        '''
        # Number of trees in the random forest
        n_estimators = [5, 20, 50, 100]
        # Maximum number of levels allowed in each decision tree
        max_depth = [10, 30, 50, 70, 90]
        # The minimum sample number to split a node
        min_samples_split = [2, 6, 10]
        # Minimum sample number that can be stored in a leaf node
        min_samples_leaf = [1, 3, 4]
        bootstrap = [True, False]  # Method used to sample data points
        # Creates a dictionary of these hyperparameters to be used in the
        # model
        parameters = {'n_estimators': n_estimators,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'bootstrap': bootstrap}
        # Initializes the model
        regressor = RandomForestRegressor()
        # Loads the hyperparameters into a grid search using the XGBoost model
        # and ensures that it is using MAE for the error
        grid_search = GridSearchCV(regressor, param_grid=parameters,
                                   scoring="neg_mean_absolute_error")
        # Fits the data to the model and the grid search
        grid_search.fit(self._train_features, self._train_labels)
        # Creates the best estimator of the most effective model using the
        # grid search
        best_est = grid_search.best_estimator_
        # Creates the feature importance from the model and plots it
        feat_importances = pd.Series(best_est.feature_importances_,
                                     index=self._train_features.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title("Random Forest Importance")
        # Gives the plot a name and saves it as a .png file
        plt.savefig("random_forest.png", bbox_inches="tight")
        # Returns the MAE of the model, while making sure that it is positive,
        # as the model is set to use a negative MAE to singify that it is
        # something to be minimized. However, it is less confusing to keep the
        # error positive
        return -grid_search.best_score_

    def decision_tree_model(self) -> float:
        '''
        This method uses the decision tree machine learning model and database
        to train and optimize the parameters for a model to predict Cy Young
        vote points. Returns the mean absolute error and makes a graph of the
        importance's in the model.
        '''
        # Initializes the model
        model = DecisionTreeRegressor()
        # Strategy to split node, either 'best' or 'random'
        splitters = ["best", "random"]
        # Maximum number of levels allowed in each decision tree
        max_depth = [3, 5, 7, 9, 11]
        # Minimum sample number that can be stored in a leaf node
        min_samples_leaf = [1, 3, 5, 7, 9]
        # The minimum weighted fraction of the sum total of weights
        # (of all the input samples) required to be at a leaf node
        min_weight_fraction_leaf = [0.1, 0.3, 0.5]
        # Creates a dictionary of these hyperparameters to be used in the
        # model
        parameters = {"splitter": splitters,
                      "max_depth": max_depth,
                      "min_samples_leaf": min_samples_leaf,
                      "min_weight_fraction_leaf": min_weight_fraction_leaf}
        # Loads the hyperparameters into a grid search using the XGBoost model
        # and ensures that it is using MAE for the error
        tuning_model = GridSearchCV(model, param_grid=parameters,
                                    scoring='neg_mean_absolute_error')
        tuning_model.fit(self._train_features, self._train_labels)
        # Creates the best estimator of the most effective model using the
        # grid search
        best_est = tuning_model.best_estimator_
        # Creates the feature importance from the model and plots it
        feat_importances = pd.Series(best_est.feature_importances_,
                                     index=self._train_features.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.title("Decision Tree Importance")
        # Gives the plot a name and saves it as a .png file
        plt.savefig("decision_tree.png", bbox_inches="tight")
        # Returns the MAE of the model, while making sure that it is positive,
        # as the model is set to use a negative MAE to singify that it is
        # something to be minimized. However, it is less confusing to keep the
        # error positive
        return -tuning_model.best_score_

    def compare_eras(self) -> None:
        '''
        This method loads in the data split into eras before 1990 and then
        1990 and after to plot and analyze the differences in correlations
        between Cy Young Award vote points and the different pitching
        statistics used in the machine learning models above. It plots these
        correlations on a 4x5 grid of charts and prints detailed information
        about the different statistical importances of these correlations,
        such as correlation coefficients, R^2, and the slope and intercepts of
        the regression lines. It also prints out the differences between these
        eras before 1990 as well as 1990 and after.
        '''
        # Creates 20 subplots in a 4x5 grid to plot the linear regression
        # plots on to
        _, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 13))
        # Flattens the axes into a 1-dimensional array
        axes = axes.ravel()
        # Creates a list that the linear regression stats for the data for
        # each pitching statistic used in the model will be appended to for
        # the analysis before 1990
        statistics_before_1990 = []
        # Loops over each feature in the data before 1990
        for i, j in enumerate(self._before_1990.columns[:-1]):
            # Initializes the linear regression model
            regressor_before_1990 = LinearRegression()
            # Fits the features from the loop to the model to be tested
            # against the vote points for the Cy Young Award, which we can
            # find from our label for the above models
            regressor_before_1990.\
                fit(self._before_1990[[j]],
                    self._before_1990[self._train_labels.name])
            # Creates a prediction from this linear regression using the fit
            # data from above
            predictions_before_1990 = regressor_before_1990.\
                predict(self._before_1990[[j]])
            # Plots this data in a scatterplot using the subplots made prior
            axes[i].scatter(self._before_1990[j],
                            self._before_1990[self._train_labels.name])
            # Creates the regression line for the linear regression
            axes[i].plot(self._before_1990[j], predictions_before_1990,
                         color="red")
            # Labels the x-axis as the given feature that is being tested
            # against the vote points and sets the title
            axes[i].set_xlabel(j)
            axes[i].set_title("Vote Points Before 1990 vs. " + j)
            # Creates the correlation coefficient from the Pearson method for
            # calculating this. We will not need the p-value that also comes
            # from this function
            r_before_1990, _ =\
                pearsonr(self._before_1990[j],
                         self._before_1990[self._train_labels.name])
            # Creates the R^2 by squaring the above value
            r2_before_1990 = r_before_1990 ** 2
            # Finds the slope from the regression model
            slope_before_1990 = regressor_before_1990.coef_[0]
            # Finds the y-intercept from the regression model
            y_intercept_before_1990 = regressor_before_1990.intercept_
            # Creates a dictionary of these statistics related to the linear
            # regression model and appends it to the above list
            statistics_before_1990.\
                append({"Pitching Stat": j,
                        "Correlation Coefficient": round(r_before_1990, 2),
                        "R^2": round(r2_before_1990, 2),
                        "Slope": round(slope_before_1990, 2),
                        "Y-Intercept": round(y_intercept_before_1990, 2)})
        # Creates a list that the linear regression stats for the data for
        # each pitching statistic used in the model will be appended to for
        # the analysis 1990 and after
        statistics_after_1989 = []
        # Loops over each feature in the data after 1990
        for m, n in enumerate(self._after_1989.columns[:-1]):
            # Initializes the linear regression model
            regressor_after_1989 = LinearRegression()
            # Fits the features from the loop to the model to be tested
            # against the vote points for the Cy Young Award, which we can
            # find from our label for the above models
            regressor_after_1989.\
                fit(self._after_1989[[n]],
                    self._after_1989[self._train_labels.name])
            # Creates a prediction from this linear regression using the fit
            # data from above
            predictions_after_1989 = regressor_after_1989.\
                predict(self._after_1989[[n]])
            # Plots this data in a scatterplot using the subplots made prior
            axes[m + 10].scatter(self._after_1989[n],
                                 self._after_1989[self._train_labels.name])
            # Creates the regression line for the linear regression
            axes[m + 10].plot(self._after_1989[n], predictions_after_1989,
                              color='red')
            # Labels the x-axis as the given feature that is being tested
            # against the vote points and sets the title
            axes[m + 10].set_xlabel(n)
            axes[m + 10].set_title("Vote Points After 1989 vs. " + n)
            # Creates the correlation coefficient from the Pearson method for
            # calculating this. We will not need the p-value that also comes
            # from this function
            r_after_1989, _ =\
                pearsonr(self._after_1989[n],
                         self._after_1989[self._train_labels.name])
            # Creates the R^2 by squaring the above value
            r2_after_1989 = r_after_1989 ** 2
            # Finds the slope from the regression model
            slope_after_1989 = regressor_after_1989.coef_[0]
            # Finds the y-intercept from the regression model
            y_intercept_after_1989 = regressor_after_1989.intercept_
            # Creates a dictionary of these statistics related to the linear
            # regression model and appends it to the above list
            statistics_after_1989.\
                append({"Pitching Stat": n,
                        "Correlation Coefficient": round(r_after_1989, 2),
                        "R^2": round(r2_after_1989, 2),
                        "Slope": round(slope_after_1989, 2),
                        "Y-Intercept": round(y_intercept_after_1989, 2)})
        # Saves all of these subplots
        plt.tight_layout()
        plt.savefig("compare_eras_chart.png", bbox_inches="tight")
        # Creates a list of the dictionaries that calculates the differences
        # between the values in the previous two dictionaries by zipping them
        # together and then looping over them using a list comprehension. It
        # Checks to see whether or not the keys of these dictionaries line up
        # and then calculates the difference between them while rounding to
        # ensure that there are no floating point errors
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
        # Prints out the linear regressions statistics for the pitchers before
        # 1990
        print("Linear regression statistics for pitchers before 1990:")
        for x in statistics_before_1990:
            # Creates a new column that gives the actual regression line
            # equation for the linear regression
            x["Line"] = str(x["Slope"]) + "X + " + str(x["Y-Intercept"])
            # Checks to see if there is the addition of negative values in
            # this line and replaces it with the subtraction of positive
            # values
            if "+ -" in x["Line"]:
                x["Line"] = x["Line"].replace("+ -", "- ")
            # Removes the columns that are redundant with the line equation
            del x["Slope"]
            del x["Y-Intercept"]
            # Turns the dictionary into a string, removes all of the quotation
            # marks, and turns the first comma into a dash
            print(str(x)[18:-1].replace("'", '').replace(",", " -", 1))
        print()
        # Prints out the linear regressions statistics for the pitchers 1990
        # and after
        print("Linear regression statistics for pitchers 1990 and after:")
        for y in statistics_after_1989:
            # Creates a new column that gives the actual regression line
            # equation for the linear regression
            y["Line"] = str(y["Slope"]) + "X + " + str(y["Y-Intercept"])
            # Checks to see if there is the addition of negative values in
            # this line and replaces it with the subtraction of positive
            # values
            if "+ -" in y["Line"]:
                y["Line"] = y["Line"].replace("+ -", "- ")
            # Removes the columns that are redundant with the line equation
            del y["Slope"]
            del y["Y-Intercept"]
            # Turns the dictionary into a string, removes all of the quotation
            # marks, and turns the first comma into a dash
            print(str(y)[18:-1].replace("'", '').replace(",", " -", 1))
        print()
        # Prints out the difference between the linear regressions statistics
        # for the pitchers before 1990 as well as those 1990 and after
        print("Differences between linear regression statistics for pitchers\
 before 1990 and after 1989:")
        for z in regression_statistics_differences:
            # Turns the dictionary into a string, removes all of the quotation
            # marks, and turns the first comma into a dash
            print(str(z)[18:-1].replace("'", '').replace(",", " -", 1))

    def starters_vs_relievers(self) -> None:
        '''
        This method loads in the data split into the different pitcher types
        of starting and relief pitchers to plot and analyze the differences in
        correlations between Cy Young Award vote points and the different
        pitching statistics used in the machine learning models above. It
        plots these correlations on a 4x5 grid of charts and prints detailed
        information about the different statistical importances of these
        correlations, such as correlation coefficients, R^2, and the slope and
        intercepts of the regression lines. It also prints out the differences
        between the different types of pitchers.
        '''
        # Creates 20 subplots in a 4x5 grid to plot the linear regression
        # plots on to
        _, axes = plt.subplots(nrows=4, ncols=5, figsize=(15, 13))
        # Flattens the axes into a 1-dimensional array
        axes = axes.ravel()
        # Creates a list that the linear regression stats for the data for
        # each pitching statistic used in the model will be appended to for
        # the analysis for starting pitchers
        statistics_starters = []
        # Loops over each feature in the data for starting pitchers
        for i, j in enumerate(self._starting_pitchers.columns[:-1]):
            # Initializes the linear regression model
            regressor_starters = LinearRegression()
            # Fits the features from the loop to the model to be tested
            # against the vote points for the Cy Young Award, which we can
            # find from our label for the above models
            regressor_starters.\
                fit(self._starting_pitchers[[j]],
                    self._starting_pitchers[self._train_labels.name])
            # Creates a prediction from this linear regression using the fit
            # data from above
            predictions_starters = regressor_starters.\
                predict(self._starting_pitchers[[j]])
            # Plots this data in a scatterplot using the subplots made prior
            axes[i].scatter(self._starting_pitchers[j],
                            self._starting_pitchers[self._train_labels.name])
            # Creates the regression line for the linear regression
            axes[i].plot(self._starting_pitchers[j],
                         predictions_starters, color="red")
            # Labels the x-axis as the given feature that is being tested
            # against the vote points and sets the title
            axes[i].set_xlabel(j)
            axes[i].set_title("Vote Points for Starters vs. " + j)
            # Creates the correlation coefficient from the Pearson method for
            # calculating this. We will not need the p-value that also comes
            # from this function
            r_starters, _ =\
                pearsonr(self._starting_pitchers[j],
                         self._starting_pitchers[self._train_labels.name])
            # Creates the R^2 by squaring the above value
            r2_starters = r_starters ** 2
            # Finds the slope from the regression model
            slope_starters = regressor_starters.coef_[0]
            # Finds the y-intercept from the regression model
            y_intercept_starters = regressor_starters.intercept_
            # Creates a dictionary of these statistics related to the linear
            # regression model and appends it to the above list
            statistics_starters.\
                append({"Pitching Stat": j,
                        "Correlation Coefficient": round(r_starters, 2),
                        "R^2": round(r2_starters, 2),
                        "Slope": round(slope_starters, 2),
                        "Y-Intercept": round(y_intercept_starters, 2)})
        # Creates a list that the linear regression stats for the data for
        # each pitching statistic used in the model will be appended to for
        # the analysis for relief pitchers
        statistics_relievers = []
        # Loops over each feature in the data for relief pitchers
        for m, n in enumerate(self._relief_pitchers.columns[:-1]):
            # Initializes the linear regression model
            regressor_relievers = LinearRegression()
            # Fits the features from the loop to the model to be tested
            # against the vote points for the Cy Young Award, which we can
            # find from our label for the above models
            regressor_relievers.\
                fit(self._relief_pitchers[[n]],
                    self._relief_pitchers[self._train_labels.name])
            # Creates a prediction from this linear regression using the fit
            # data from above
            predictions_relievers = regressor_relievers.\
                predict(self._relief_pitchers[[n]])
            # Plots this data in a scatterplot using the subplots made prior
            axes[m + 10].\
                scatter(self._relief_pitchers[n],
                        self._relief_pitchers[self._train_labels.name])
            # Creates the regression line for the linear regression
            axes[m + 10].plot(self._relief_pitchers[n],
                              predictions_relievers, color='red')
            # Labels the x-axis as the given feature that is being tested
            # against the vote points and sets the title
            axes[m + 10].set_xlabel(n)
            axes[m + 10].set_title("Vote Points for Relievers vs. " + n)
            # Creates the correlation coefficient from the Pearson method for
            # calculating this. We will not need the p-value that also comes
            # from this function
            r_relievers, _ =\
                pearsonr(self._relief_pitchers[n],
                         self._relief_pitchers[self._train_labels.name])
            # Creates the R^2 by squaring the above value
            r2_relievers = r_relievers ** 2
            # Finds the slope from the regression model
            slope_relievers = regressor_relievers.coef_[0]
            # Finds the y-intercept from the regression model
            y_intercept_relievers = regressor_relievers.intercept_
            # Creates a dictionary of these statistics related to the linear
            # regression model and appends it to the above list
            statistics_relievers.\
                append({"Pitching Stat": n,
                        "Correlation Coefficient": round(r_relievers, 2),
                        "R^2": round(r2_relievers, 2),
                        "Slope": round(slope_relievers, 2),
                        "Y-Intercept": round(y_intercept_relievers, 2)})
        # Saves all of these subplots
        plt.tight_layout()
        plt.savefig("starters_vs_relievers_chart.png", bbox_inches="tight")
        # Creates a list of the dictionaries that calculates the differences
        # between the values in the previous two dictionaries by zipping them
        # together and then looping over them using a list comprehension. It
        # Checks to see whether or not the keys of these dictionaries line up
        # and then calculates the difference between them while rounding to
        # ensure that there are no floating point errors
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
        # Prints out the linear regressions statistics for the starting
        # pitchers
        print("Linear regression statistics for starting pitchers:")
        for x in statistics_starters:
            # Creates a new column that gives the actual regression line
            # equation for the linear regression
            x["Line"] = str(x["Slope"]) + "X + " + str(x["Y-Intercept"])
            # Checks to see if there is the addition of negative values in
            # this line and replaces it with the subtraction of positive
            # values
            if "+ -" in x["Line"]:
                x["Line"] = x["Line"].replace("+ -", "- ")
            # Removes the columns that are redundant with the line equation
            del x["Slope"]
            del x["Y-Intercept"]
            # Turns the dictionary into a string, removes all of the quotation
            # marks, and turns the first comma into a dash
            print(str(x)[18:-1].replace("'", '').replace(",", " -", 1))
        print()
        # Prints out the linear regressions statistics for the relief
        # pitchers
        print("Linear regression statistics for relief pitchers:")
        for y in statistics_relievers:
            # Creates a new column that gives the actual regression line
            # equation for the linear regression
            y["Line"] = str(y["Slope"]) + "X + " + str(y["Y-Intercept"])
            # Checks to see if there is the addition of negative values in
            # this line and replaces it with the subtraction of positive
            # values
            if "+ -" in y["Line"]:
                y["Line"] = y["Line"].replace("+ -", "- ")
            # Removes the columns that are redundant with the line equation
            del y["Slope"]
            del y["Y-Intercept"]
            # Turns the dictionary into a string, removes all of the quotation
            # marks, and turns the first comma into a dash
            print(str(y)[18:-1].replace("'", '').replace(",", " -", 1))
        print()
        # Prints out the differences in the linear regressions statistics
        # between starting pitchers and relief pitchers
        print("Differences between linear regression statistics for starters\
 and relievers:")
        for z in starters_vs_relievers_statistics_differences:
            # Turns the dictionary into a string, removes all of the quotation
            # marks, and turns the first comma into a dash
            print(str(z)[18:-1].replace("'", '').replace(",", " -", 1))


def main():
    start_year = 1956
    end_year = 2016
    id_filename = "players.csv"
    pitching = PitchingModel(start_year, end_year, id_filename)
    print(pitching.xgboost_model())
    print(pitching.random_forest_model())
    print(pitching.decision_tree_model())
    print(pitching.compare_eras())
    print(pitching.starters_vs_relievers())


if __name__ == '__main__':
    main()
