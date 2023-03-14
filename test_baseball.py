'''
Aidan Murphy, Luke VanHouten
CSE 163 section AH
13 March 2023

This program implements the testing files to test our different methods from
our two classes PitchingData and PitchingModel to ensure that the loading in
of data for our model to predict Cy Young Award vote points based on the
pitching statistics of top-tier pitchers in Major League Baseball as well as
the the output of the model itself do not return errors.
'''

import math
from baseball_data import PitchingData
from baseball_model import PitchingModel
from cse163_utils import assert_equals


def test_data_for_model(id_filename: str) -> None:
    '''
    This testing function ensures that the shapes of the data methods to be
    used for the machine learning part of the project (question 1) are correct.
    I went to baseball-reference.com and checked to make sure that the length
    of each dataframe is the same as the amount of pitchers who receieved
    votes in 2016. This year was chosen as it is very simple to run and also
    ensures that running the same year for both the start and end dates works
    as planned with the data. It takes a string input for the file name for
    the players.csv file that joins the data in the data file as people using
    this may have different file paths.
    '''
    # Loads in the data from the PitchingData class
    pitching_data = PitchingData(2016, 2016, id_filename)
    merged_data = pitching_data.view_merged_data().shape
    # Checks to see that the merged data matches the expected output
    assert_equals((21, 17), merged_data)
    train_features = pitching_data.train_features().shape
    train_labels = pitching_data.train_labels().shape
    test_features = pitching_data.test_features().shape
    test_labels = pitching_data.test_labels().shape
    # Checks to see if the train test split of the data actually matches the
    # 20% ratio of test data to train data
    assert_equals(math.floor(0.8 * merged_data[0]), train_features[0])
    assert_equals(math.floor(0.8 * merged_data[0]), train_labels[0])
    assert_equals(math.ceil(0.2 * merged_data[0]), test_features[0])
    assert_equals(math.ceil(0.2 * merged_data[0]), test_labels[0])
    # Checks to see that the shape of these add up to each other
    assert_equals(train_features[0] + test_features[0],
                  train_labels[0] + test_labels[0])
    # Ensures the correct amount of columns in the data
    assert_equals(10, train_features[1])
    assert_equals(10, test_features[1])


def test_data_for_regression(id_filename: str) -> None:
    '''
    This testing function ensures that the shapes of the data methods to be
    used for the linear regressions questions (2 and 3) of the project are
    correct. I went to baseball-reference.com and checked to make sure that
    the length of each dataframe is the same as the amount of pitchers who
    receieved votes between 1988 and 1991. These years were chosen in order to
    test the split before 1990 and 1990 and after.  It takes a string input
    for the file name for the players.csv file that joins the data in the data
    file as people using this may have different file paths.
    '''
    # Loads in the data from the PitchingData class
    pitching_data = PitchingData(1988, 1991, id_filename)
    before_1990 = pitching_data.before_1990().shape
    after_1989 = pitching_data.after_1989().shape
    starting_pitchers = pitching_data.starting_pitchers().shape
    relief_pitchers = pitching_data.relief_pitchers().shape
    # Checks to see if the actual values of the shape of the data match the
    # amount of Cy Young Award vote-getters for those years in each era
    assert_equals(28, before_1990[0])
    assert_equals(30, after_1989[0])
    assert_equals(45, starting_pitchers[0])
    assert_equals(13, relief_pitchers[0])
    # Checks to see that the shape of these add up to each other
    assert_equals(before_1990[0] + after_1989[0],
                  starting_pitchers[0] + relief_pitchers[0])
    # Ensures the correct amount of columns in the data
    assert_equals(11, before_1990[1])
    assert_equals(11, after_1989[1])
    assert_equals(11, starting_pitchers[1])
    assert_equals(11, relief_pitchers[1])


def test_models_accuracy(id_filename: str) -> None:
    '''
    This testing function ensures that the error value that comes from all 3
    of the models used in our analysis is within the actual possible points
    range there is for the Cy Young Award vote points, which is between 0 and
    224, which is a standardized value based on the maximum amount of points
    possible for a year in the history of the award. If the error is somewhere
    between these numbers, it will tell us that our model makes sense,
    although it doesn't say anything about how good the error value is. It
    will assert it as being equal if the value is in the range as that will
    return a boolean True.
    '''
    # Checks to see whether or not the errors from each of the models fits
    # within the acceptible range of the values for these vote points from the
    # testing data
    assert_equals(True, 0 < PitchingModel(1956, 2016,
                                          id_filename).xgboost_model() < 224)
    assert_equals(True,
                  0 < PitchingModel(1956, 2016,
                                    id_filename).random_forest_model() < 224)
    assert_equals(True,
                  0 < PitchingModel(1956, 2016,
                                    id_filename).decision_tree_model() < 224)


def main():
    id_filename = "players.csv"
    test_data_for_model(id_filename)
    test_data_for_regression(id_filename)
    test_models_accuracy(id_filename)


if __name__ == '__main__':
    main()
