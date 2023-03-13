'''
This file contains tests for the methods in the PitchingData and PitchingModel
classes
'''
from baseball_model import PitchingModel
from baseball_data import PitchingData


def test_merge(pd) -> None:
    '''
    Takes a PitchingData class as a parameter. Tests the merge_data method,
    printing the first 10 rows of data
    '''
    print('merged data: ')
    print(PitchingData.view_merged_data(pd).head(n=10))


def test_train_features(pd) -> None:
    print('training features: ')
    print(PitchingData.train_features(pd))


def test_train_labels(pd) -> None:
    print('training labels: ')
    print(PitchingData.train_labels(pd))


def test_test_features(pd) -> None:
    print('test features: ')
    print(PitchingData.test_features(pd))


def test_test_labels(pd) -> None:
    print('test labels: ')
    print(PitchingData.test_labels(pd))


def test_before_1990(pd) -> None:
    '''
    Takes a PitchingData class as a parameter. Tests the before_1990 method,
    printing the first 5 rows of data
    '''
    print('before 1990: ')
    print(PitchingData.before_1990(pd).head(n=5))


def test_after_1989(pd) -> None:
    '''
    Takes a PitchingData class as a parameter. Tests the after_1989 method,
    printing the first 5 rows of data
    '''
    print('after 1989: ')
    print(PitchingData.after_1989(pd).head(n=5))


def test_starting_pitchers(pd) -> None:
    '''
    Takes a PitchingData class as a parameter. Tests the starting_pitchers
    method, printing the first 5 rows of data
    '''
    print('starting pitchers: ')
    print(PitchingData.starting_pitchers(pd).head(n=5))


def test_relief_pitchers(pd) -> None:
    '''
    Takes a PitchingData class as a parameter. Tests the relief_pitchers
    method, printing the first 5 rows of data
    '''
    print('relief pitchers: ')
    print(PitchingData.relief_pitchers(pd).head(n=5))


def test_accuracy_tree(pm) -> None:
    '''
    Takes a PitchingModel class as a parameter. Tests the decision tree
    model in decision_tree_model, printing the returned error
    '''
    print('tree: ' + str(PitchingModel.decision_tree_model(pm)))


def test_acc_boost(pm) -> None:
    '''
    Takes a PitchingModel class as a parameter. Tests the xgboost
    model in xgboost_model, printing the returned error
    '''
    print('xgboost: ' + str(PitchingModel.xgboost_model(pm)))


def test_acc_forest(pm) -> None:
    '''
    Takes a PitchingModel class as a parameter. Tests the random forest
    model in random_forest_model, printing the returned error
    '''
    print('random forest: ' + str(PitchingModel.random_forest_model(pm)))


def test_eras(pm) -> None:
    '''
    Takes a PitchingModel class as a parameter. Runs the compare_eras
    method, creating graphs showing the differences in era voting
    '''
    PitchingModel.compare_eras(pm)


def test_starters_vs_relievers(pm) -> None:
    '''
    Takes a PitchingModel class as a parameter. Runs the starters_vs_relievers
    method, creating graphs showing the differences in voting between starters
    and relievers
    '''
    PitchingModel.starters_vs_relievers(pm)


def main() -> None:
    '''
    Creates PitchingData and PitchingModel objects and calls the different
    testing methods
    '''
    pd = PitchingData(start_year=2006, end_year=2010)
    pm = PitchingModel(start_year=2006, end_year=2010)
    test_merge(pd)
    test_train_labels(pd)
    test_train_features(pd)
    test_test_labels(pd)
    test_test_features(pd)
    test_before_1990(pd)
    test_after_1989(pd)
    test_starting_pitchers(pd)
    test_relief_pitchers(pd)
    test_accuracy_tree(pm)
    test_acc_forest(pm)
    test_acc_boost(pm)
    test_eras(pm)
    test_starters_vs_relievers(pm)


if __name__ == '__main__':
    main()
