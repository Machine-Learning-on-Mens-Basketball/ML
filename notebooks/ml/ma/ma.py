"""This module contains functions used in the jupyter notebooks"""


__author__ = "Sam Kasbawala"
__credits__ = ["Sam Kasbawala"]

__licence__ = "BSD"
__maintainer__ = "Sam Kasbawala"
__email__ = "samarth.kasbawala@uconn.edu"
__status__ = "Development"


import os
import pandas as pd

from tqdm import tqdm


def get_training_data(span, identifiers, team_stats_dir):
    """Returns a DataFrame that can be used for training a model. Serves as a wrapper
    function for the compute_ma and merge_ma functions as both would generally be called
    sequentially.

    Args:
        span (int): The number of games used in the moving averages calculations.
        identifiers (list): A list of columns that aren't to be used in calculations.
        team_stats_dir (string): A string to where the team stats are stroed for each
            team. Assumes each team has their own csv file for stats.
        identifiers ([type]): [description]

    Returns:
        pandas.DataFrame: DataFrame where each row contains the outcome of the game and
            stats of the participating teams going into the game.
    """

    print("Building training data...")
    team_mas = compute_ma(span, identifiers, team_stats_dir)
    return merge_ma(team_mas, identifiers)


def compute_ma(span, identifiers, team_stats_dir):
    """This function computes the moving averages for each team.

    Args:
        span (int): The number of games used in the moving averages calculations.
        identifiers (list): A list of columns that aren't to be used in calculations.
        team_stats_dir (string): A string to where the team stats are stroed for each
            team. Assumes each team has their own csv file for stats.
        identifiers (list): A list of strings that uniquely identify a game.

    Returns:
        dict: Key, value pairs where each key is a string of the school and the value is
            the dataframe of the of moving averages.
    """

    # Dictionary to store output
    dfs = dict()

    # Loop through each file (each team has their own file)
    for team_file in tqdm(
        os.listdir(team_stats_dir), unit="Teams", desc="Calculating...".ljust(20)
    ):

        # Assumes team name is "{COLLEGE_NAME}.csv"
        team_name = team_file[:-4]

        # Load the csv
        team_stats = pd.read_csv(os.path.join(team_stats_dir, team_file)).sort_values(
            by="date"
        )

        # Compute moving averages for the appropriate columns
        for col in team_stats.columns:
            if col in identifiers:
                continue

            # Simple moving average
            team_stats[f"{col}_sma"] = (
                team_stats.loc[:, col].rolling(window=span).mean()
            )
            team_stats[f"{col}_sma"] = team_stats[f"{col}_sma"].shift(1)

            # Cumulative moving average
            team_stats[f"{col}_cma"] = (
                team_stats.loc[:, col].expanding(min_periods=span).mean()
            )
            team_stats[f"{col}_cma"] = team_stats[f"{col}_cma"].shift(1)

            # Exponential moving average
            team_stats[f"{col}_ema"] = (
                team_stats.loc[:, col].ewm(span=span, adjust=False).mean()
            )
            team_stats[f"{col}_ema"] = team_stats[f"{col}_ema"].shift(1)

        # Drop any rows with NULL values and add the CSV to the dict
        team_stats.dropna(inplace=True)
        team_stats.drop_duplicates(inplace=True)
        dfs[team_name] = team_stats

    return dfs


def merge_ma(teams_mas, identifiers):
    """Merges the team moving averages such that the data can be trained on.

    Args:
        teams_mas (dict): Key, value pairs where each key is a string of the school and
            the value is the dataframe of the of moving averages.
        identifiers (list): A list of strings that uniquely identify a game.

    Returns:
        pandas.DataFrame: DataFrame where each row contains the outcome of the game and
            stats of the participating teams going into the game.

    NOTE: This function will spit out a performance warning, this is not an error in the
    code however. There is a issue on the pandas_ta repo that talks about this which can
    be found here: https://github.com/twopirllc/pandas-ta/issues/340. To surpress this
    warning, add the following to the code:
        from warnings import simplefilter
        simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
    """

    # Arrays to hold home and aways dfs
    home_games = []
    away_games = []

    # Loop through teams
    for team, team_stats in tqdm(
        teams_mas.items(), unit="Teams", desc="Merging...".ljust(20)
    ):

        # Get all the away games
        away = team_stats.loc[team_stats["away"] == team].copy()
        away.dropna(inplace=True)
        away.drop_duplicates(inplace=True)
        away.columns = [
            col if col in identifiers else "away_" + col for col in list(away.columns)
        ]
        away_games.append(away)

        # Get all the home games
        home = team_stats.loc[team_stats["home"] == team].copy()
        home.dropna(inplace=True)
        home.drop_duplicates(inplace=True)
        home.columns = [
            col if col in identifiers else "home_" + col for col in list(home.columns)
        ]
        home_games.append(home)

    # Return a merged dataframe with all matchups
    return (
        pd.merge(pd.concat(away_games), pd.concat(home_games), on=identifiers)
        .sort_values(by="date")
        .reset_index(drop=True)
    )
