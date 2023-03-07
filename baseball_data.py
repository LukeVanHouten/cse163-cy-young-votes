import pybaseball as pb
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split

warnings.simplefilter(action="ignore", category=FutureWarning)
pd.set_option("display.max_columns", None)


class PitchingData:
    def __init__(self) -> None:
        awards = pb.lahman.awards_share_players()
        cy_young_votes = awards[awards["awardID"] == "Cy Young"]
        cy_young_votes = cy_young_votes.loc[:, ["yearID", "playerID",
                                                "pointsWon", "pointsMax"]]
        cy_young_votes["points_share"] = cy_young_votes["pointsWon"] /\
            cy_young_votes["pointsMax"]
        cy_young_votes["standardized_points"] =\
            cy_young_votes["points_share"] * max(cy_young_votes["pointsMax"])

        pitching_stats_1956_1985 = pb.pitching_stats(1956, 1985, qual=30)
        pitching_stats_1986_2016 = pb.pitching_stats(1986, 2016, qual=30)
        pitching_stats = pd.concat([pitching_stats_1956_1985,
                                    pitching_stats_1986_2016])
        pitching_stats = pitching_stats.loc[:, ["Name", "Season", "IDfg", "W",
                                                "L", "G", "GS", "ERA", "WHIP",
                                                "FIP", "IP", "SO", "K%",
                                                "K/9", "HR/9", "BB/9", "WAR"]]
        pitching_stats["win_percentage"] = pitching_stats["W"] /\
            pitching_stats["L"]

        player_ids = pd.read_csv("players.csv", encoding="latin-1")
        bad_ids = ["richaj.01", "garcifr03", "santajo02", "sabatc.01",
                   "dicker.01"]
        good_ids = ["richajr01", "garcifr02", "santajo01", "sabatcc01",
                    "dickera01"]
        for j, k in zip(bad_ids, good_ids):
            player_ids.loc[player_ids["key_bbref"] == j, "key_bbref"] = k
        player_ids = player_ids.loc[:, ["key_bbref", "key_fangraphs"]]

        cy_young_ids = cy_young_votes.merge(player_ids, left_on="playerID",
                                            right_on="key_bbref")

        merged_data = pitching_stats.merge(cy_young_ids,
                                           left_on=["IDfg", "Season"],
                                           right_on=["key_fangraphs",
                                                     "yearID"])
        self._cy_young_vote_getters: pd.DataFrame =\
            merged_data.drop(columns=["IDfg", "yearID", "playerID",
                                      "pointsWon", "pointsMax", "key_bbref",
                                      "points_share", "key_fangraphs"])

        features = self._cy_young_vote_getters.\
            loc[:, ~self._cy_young_vote_getters.
                columns.isin(["Name", "Season", "L", "G", "GS",
                              "standardized_points"])]
        labels = self._cy_young_vote_getters["standardized_points"]

        self._train_features, self._test_features, self._train_labels,\
            self._test_labels = train_test_split(features, labels,
                                                 test_size=0.2, shuffle=True)

    def view_merged_data(self) -> pd.DataFrame:
        return self._cy_young_vote_getters

    def train_features(self) -> pd.DataFrame:
        return self._train_features

    def train_labels(self) -> pd.DataFrame:
        return self._train_labels

    def test_features(self) -> pd.DataFrame:
        return self._test_features

    def test_labels(self) -> pd.DataFrame:
        return self._test_labels


PitchingData().view_merged_data()
