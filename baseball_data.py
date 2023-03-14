'''
Aidan Murphy, Luke VanHouten
CSE 163 section AH
13 March 2023

This program implements the class PitchingData to initialize the data used in
our model to predict Cy Young Award vote points based on the pitching
statistics of top-tier pitchers in Major League Baseball
'''

import pybaseball as pb
import pandas as pd
from sklearn.model_selection import train_test_split


class PitchingData:
    '''
    This class loads in the data used to predict MLB Cy Young Award votes for
    top-tier pitchers based on their stats for a given season. The class will
    then return the data split into training and testing features and labels
    for use in a machine learning model.
    '''
    def __init__(self, start_year: int, end_year: int,
                 id_filename: str) -> None:
        '''
        This method initializes the PitchingData class in order to load in
        data for stats for baseball pitchers, awards data to find the vote
        getters for the MLB Cy Young Award for best pitching. The data is then
        cleaned and split into training and testing features and labels for
        use in the ML model. The data for the pitcher stats is from FanGraphs,
        the data for the awards is from the Lahman baseball database, and the
        data for player IDs is from the Chadwick Bureau of baseball players.
        The latter data is initialized with a string filename that also has to
        be inputted along with the dates.
        '''
        # Loads in the awards data from the Lahman baseball database method
        # from the bypaseball module
        awards = pb.lahman.awards_share_players()
        # Filters the awards to be only for the Cy Young award, and selects
        # the columns for the player, year, and amount of points won as well
        # as the amount that are possible to win
        cy_young_votes = awards[awards["awardID"] == "Cy Young"]
        cy_young_votes = cy_young_votes.loc[:, ["yearID", "playerID",
                                                "pointsWon", "pointsMax"]]
        # Creates a new column that is the percentage of the points share each
        # player for a given year got for the Cy Young award
        cy_young_votes["points_share"] = cy_young_votes["pointsWon"] /\
            cy_young_votes["pointsMax"]
        # Creates a new column that standardizes this points share to the
        # maximum amount of points ever possible for a Cy Young award
        cy_young_votes["standardized_points"] =\
            cy_young_votes["points_share"] * max(cy_young_votes["pointsMax"])
        # Because of how many years there are for the Cy Young award, we need
        # to parse through the data in two separate calls. Here we create a
        # value for the middle year of the range by find the difference
        # between the start and end years, dividing this in half, rounding it
        # up to the nearest year, and subtracting it from the end year
        middle_year = end_year - round((end_year - start_year) / 2)
        # Calls the pybaseball method to get all of the FanGraphs pitching
        # stats per player for the first half of our date range. The keyword
        # argument here represents the minimum number of innings pitched in
        # order for a player to be qualified for the dataset. This value is 38
        # in order to reduce the time it takes to access the data, as this is
        # the least amount of innings that any player who has receieved a Cy
        # Young award vote has pitched.
        pitching_stats_1st = pb.pitching_stats(start_year, middle_year,
                                               qual=38)
        # Checks to see if the start year is less than the end year, and calls
        # the method to access the FanGraphs data again for the second half of
        # the year range. Gives an empty Pandas dataframe if the year values
        # are the same (meaning that only one year of data is to be returned),
        # as we only need to call the method once to do this.
        if start_year < end_year:
            pitching_stats_2nd = pb.pitching_stats(middle_year + 1, end_year,
                                                   qual=38)
        else:
            pitching_stats_2nd = pd.DataFrame()
        # Combines the two dataframes from the pitching data method calls into
        # one Pandas dataframe
        pitching_stats = pd.concat([pitching_stats_1st, pitching_stats_2nd])
        pitching_stats = pitching_stats.loc[:, ["Name", "Season", "IDfg", "W",
                                                "L", "G", "GS", "ERA", "WHIP",
                                                "FIP", "IP", "SO", "K%",
                                                "K/9", "HR/9", "BB/9", "WAR"]]
        # Access the data for the player IDs. Because many players are from
        # Latin America and have accents in their names, we encode this data
        # in the latin-1 encoding in order to prevent any errors
        player_ids = pd.read_csv(id_filename, encoding="latin-1")
        # Five players in this data do not have baseball-reference IDs that
        # match up with those from the Lahman database. These players are: J.R
        # Richard, Freddy Garcia, Johan Santana, CC Sabathia, and R.A. Dickey.
        # Their IDs have been replaced with the ones found on each pitcher's
        # respective Baseball-Reference page
        bad_ids = ["richaj.01", "garcifr03", "santajo02", "sabatc.01",
                   "dicker.01"]
        good_ids = ["richajr01", "garcifr02", "santajo01", "sabatcc01",
                    "dickera01"]
        for j, k in zip(bad_ids, good_ids):
            player_ids.loc[player_ids["key_bbref"] == j, "key_bbref"] = k
        # Selects only the Baseball-reference and FanGraphs IDs. The former is
        # used by the Lahman database, and the latter is used by the FanGraphs
        # pitching data
        player_ids = player_ids.loc[:, ["key_bbref", "key_fangraphs"]]
        # Merges these IDs to the Cy Young award data by the
        # Baseball-Reference IDs
        cy_young_ids = cy_young_votes.merge(player_ids, left_on="playerID",
                                            right_on="key_bbref")
        # Merges the Cy Young award data to the FanGraphs pitching data by the
        # Fangraphs IDs
        merged_data = pitching_stats.merge(cy_young_ids,
                                           left_on=["IDfg", "Season"],
                                           right_on=["key_fangraphs",
                                                     "yearID"])
        # Removes columns that are not necessary for the analysis portion
        self._cy_young_vote_getters: pd.DataFrame =\
            merged_data.drop(columns=["IDfg", "yearID", "playerID",
                                      "pointsWon", "pointsMax", "key_bbref",
                                      "points_share", "key_fangraphs"])
        # Selects all of the columns that are features, which will be every
        # pitching stat excluding losses as well as the feature, the points
        # for the Cy Young award
        features = self._cy_young_vote_getters.\
            drop(columns=["Name", "Season", "L", "G", "GS", "IP",
                          "standardized_points"])
        # Selects the column that is the label, which are the points for the
        # Cy Young award
        labels = self._cy_young_vote_getters["standardized_points"]
        # Splits the testing and training labels, with 20% of the data being
        # testing data, with this being randomized
        self._train_features, self._test_features, self._train_labels,\
            self._test_labels = train_test_split(features, labels,
                                                 test_size=0.2, shuffle=True)

    def view_merged_data(self) -> pd.DataFrame:
        '''
        This method returns all of the data to be used in the model, for the
        purposes of perusing the data for information and incites.
        '''
        return self._cy_young_vote_getters

    def train_features(self) -> pd.DataFrame:
        '''
        This method returns the features for our training data for our model.
        '''
        return self._train_features

    def train_labels(self) -> pd.DataFrame:
        '''
        This method returns the label for our training data for our model,
        which is the amount of Cy Young points awarded to a given pitcher.
        This is standardized to 224 points, which is the maximum awarded for
        any year. Different points are awarded to different votes; a first
        place vote will give more points than a fifth place vote.

        '''
        return self._train_labels

    def test_features(self) -> pd.DataFrame:
        '''
        This method returns the features for our testing data for our model.
        '''
        return self._test_features

    def test_labels(self) -> pd.DataFrame:
        '''
        This method returns the label for our testing data for our model.
        '''
        return self._test_labels

    def before_1990(self):
        '''
        This method returns all vote getters in the year range prior to the
        year 1990.
        '''
        # We filter out the same rows as we did for the features for the
        # model, although we keep the points for our linear regression.
        return self._cy_young_vote_getters[self._cy_young_vote_getters
                                           ["Season"] < 1990].\
            drop(columns=["Name", "Season", "L", "G", "GS", "IP"])

    def after_1989(self):
        '''
        This method returns all vote getters in the year range after and
        including the year 1990.
        '''
        # We filter out the same rows as we did for the features for the
        # model, although we keep the points for our linear regression.
        return self._cy_young_vote_getters[self._cy_young_vote_getters
                                           ["Season"] >= 1990].\
            drop(columns=["Name", "Season", "L", "G", "GS", "IP"])

    def starting_pitchers(self):
        '''
        This method returns the data for all starting pitchers in the date
        range. A starting pitcher is a pitcher whose games played includes a
        substantial amount of starts.
        '''
        # We filter out the same rows as we did for the features for the
        # model, although we keep the points for our linear regression.
        return self._cy_young_vote_getters[self._cy_young_vote_getters["GS"]
                                           > 3].drop(columns=["Name", "IP",
                                                              "Season", "L",
                                                              "G", "GS"])

    def relief_pitchers(self):
        '''
        This method returns the data for all relief pitchers in the date
        range. A relief pitcher is a pitcher who typically comes in later in
        the game as opposed to starting it. Because there are times when
        relief pitchers may end up starting a game for one inning, we have
        defined relief pitchers to include pitchers who have at most 3 starts.
        '''
        # We filter out the same rows as we did for the features for the
        # model, although we keep the points for our linear regression.
        return self._cy_young_vote_getters[self._cy_young_vote_getters["GS"]
                                           <= 3].drop(columns=["Name", "IP",
                                                               "Season", "L",
                                                               "G", "GS"])
