from baseball_model import PitchingModel
from baseball_data import PitchingData


def test_merge(pd):
    print('merged data: ')
    print(PitchingData.view_merged_data(pd).head(n=10))


def test_train_features(pd):
    print('training features: ')
    print(PitchingData.train_features(pd))


def test_train_labels(pd):
    print('training labels: ')
    print(PitchingData.train_labels(pd))


def test_test_features(pd):
    print('test features: ')
    print(PitchingData.test_features(pd))


def test_test_labels(pd):
    print('test labels: ')
    print(PitchingData.test_labels(pd))


def test_before_1990(pd):
    print('before 1990: ')
    print(PitchingData.before_1990(pd).head(n=5))


def test_after_1989(pd):
    print('after 1989: ')
    print(PitchingData.after_1989(pd).head(n=5))


def test_starting_pitchers(pd):
    print('starting pitchers: ')
    print(PitchingData.starting_pitchers(pd).head(n=5))


def test_relief_pitchers(pd):
    print('relief pitchers: ')
    print(PitchingData.relief_pitchers(pd).head(n=5))


def test_accuracy_tree(pm):
    print('tree: ' + str(PitchingModel.decision_tree_model(pm)))


def test_acc_boost(pm):
    print('xgboost: ' + str(PitchingModel.xgboost_model(pm)))


def test_acc_forest(pm):
    print('random forest: ' + str(PitchingModel.random_forest_model(pm)))


def test_eras(pm):
    PitchingModel.compare_eras(pm)


def main():
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
    #test_accuracy_tree(pm)
    #test_acc_forest(pm)
    #test_acc_boost(pm)
    #test_eras(pm)


if __name__ == '__main__':
    main()
