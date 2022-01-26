import sqlite3
import pandas as pd
import tensorflow
import keras
import numpy as np
import sklearn
from sklearn import linear_model
from plot_linear_home_goals_model import plot_linear_regression_space
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style
from mpl_toolkits import mplot3d
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from  mlxtend.plotting import plot_decision_regions

DATABASE_PATH = "data/database.sqlite"


# accepts database path and SQL query
def query_database(database_path: str, query: str):
    # Create a SQL connection to our SQLite database
    con = sqlite3.connect(database_path)
    cur = con.cursor()

    # The result of a "cursor.execute" can be iterated over by row
    q_result = cur.execute(query)
    df = pd.DataFrame(q_result.fetchall())
    names = list(map(lambda x: x[0], q_result.description))
    df.columns = names
    con.close()

    return df

# Generate linear model using train test split
def linear_regression_sklearn(data_points: pd.DataFrame, y_value_col: str):
    x = np.array(data_points.drop(y_value_col, axis=1))
    y = np.array(data_points[y_value_col])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.1)

    linear_home_goal = linear_model.LinearRegression()

    linear_home_goal.fit(x_train, y_train)

    accuracy = linear_home_goal.score(x_test, y_test)

    return accuracy, linear_home_goal.coef_, linear_home_goal.intercept_

# generate knn model and plot visualization
def knn_sklearn(data_points: pd.DataFrame, y_value_col: str, k: int):
    x = np.array(data_points.drop(y_value_col, axis=1))
    y = np.array(data_points[y_value_col])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.1)

    knn_goal_difference = KNeighborsClassifier(n_neighbors=k)

    knn_goal_difference.fit(x_train, y_train)

    accuracy = knn_goal_difference.score(x_test, y_test)

    plot_decision_regions(x_train, y_train, clf=knn_goal_difference, legend=2)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Knn with K=+' + str(k))
    plt.show()
    return accuracy

# Generate svm model using train test split
def svm_sklearn(data_points: pd.DataFrame, y_value_col: str):
    x = np.array(data_points.drop(y_value_col, axis=1))
    y = np.array(data_points[y_value_col])
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=.1)

    svm_clf = svm.SVC(kernel='rbf')
    svm_clf.fit(x_train, y_train)

    y_pred = svm_clf.predict(x_test)

    svm_accuracy = metrics.accuracy_score(y_test, y_pred)
    return svm_accuracy


# accept a list of player IDs and return their average rating from FIFA video game
def get_team_rating(list_player_ids, player_attributes, players):
    list_player_ids.dropna(inplace=True)

    fifa_api_ids = players.loc[list_player_ids, 'player_fifa_api_id']
    fifa_api_ids.dropna(inplace=True)

    df_team_attributes = player_attributes.loc[fifa_api_ids, ['overall_rating']]

    return df_team_attributes["overall_rating"].max()

# Run data analysis and display corresponding visualizations
def run_analysis():
    df = query_database(DATABASE_PATH,
                        f"""SELECT m.date,
                                                            h_team.team_short_name AS HomeTeam,
                                                            m.home_team_api_id,
                                                            a_team.team_short_name AS AwayTeam,
                                                            m.away_team_api_id,
                                                            m.home_team_goal,
                                                            m.away_team_goal,
                                                            m.home_player_1,
                                                            m.home_player_2,
                                                            m.home_player_3,
                                                            m.home_player_4,
                                                            m.home_player_5,
                                                            m.home_player_6,
                                                            m.home_player_7,
                                                            m.home_player_8,
                                                            m.home_player_9,
                                                            m.home_player_10,
                                                            m.home_player_11,
                                                            m.away_player_1,
                                                            m.away_player_2,
                                                            m.away_player_3,
                                                            m.away_player_4,
                                                            m.away_player_5,
                                                            m.away_player_6,
                                                            m.away_player_7,
                                                            m.away_player_8,
                                                            m.away_player_9,
                                                            m.away_player_10,
                                                            m.away_player_11

                                                        FROM Match m LEFT JOIN Team h_team ON h_team.team_api_id = m.home_team_api_id
                                                                            LEFT JOIN Team a_team ON a_team.team_api_id = m.away_team_api_id
                                                   WHERE country_id = 1729;"""
                        )

    season = '2015/2016'
    season_date = '2015-08-01'
    df = df[df['date']>=season_date]

    players = query_database(DATABASE_PATH, f"""SELECT * FROM Player;""")
    players.set_index('player_api_id', inplace=True)



    player_attributes = query_database(DATABASE_PATH, f"""SELECT * FROM Player_Attributes;""")
    player_attributes.set_index('player_fifa_api_id', inplace=True)

    print(player_attributes.columns)




    df['home_fifa_overall'] = df.apply(lambda x: get_team_rating(x[['home_player_1'
        ,'home_player_2'
        ,'home_player_3'
        , 'home_player_4'
        , 'home_player_5'
        , 'home_player_6'
        , 'home_player_7'
        , 'home_player_8'
        , 'home_player_9'
        , 'home_player_10'
        , 'home_player_11']]
                                                                 , player_attributes
                                                                 , players), axis=1)

    df['away_fifa_overall'] = df.apply(lambda x: get_team_rating(x[['away_player_1'
        ,'away_player_2'
        ,'away_player_3'
        , 'away_player_4'
        , 'away_player_5'
        , 'away_player_6'
        , 'away_player_7'
        , 'away_player_8'
        , 'away_player_9'
        , 'away_player_10'
        , 'away_player_11']]
                                                                 , player_attributes
                                                                 , players), axis=1)



    df['home_team_goal_differential'] = df['home_team_goal'] - df['away_team_goal']

    # break up the data into X and Y vals
    df_predictions = df[['home_team_goal_differential'
        ,'home_fifa_overall'
        ,'away_fifa_overall']]

    print(df_predictions)
    accuracy, coefficients, intercept = linear_regression_sklearn(df_predictions, 'home_team_goal_differential')

    print("Linear regression model accuracy is:: " + str(accuracy))
    print('coefficient: /n', coefficients)
    print('intercept: /n', intercept)

    # Linear model is not a great fit.
    knn_accuracy = knn_sklearn(df_predictions, 'home_team_goal_differential', 5)
    print("KNN model accuracy is:: " + str(knn_accuracy))

    plot_linear_regression_space(df_predictions)

    svm_accuracy = svm_sklearn(df_predictions, 'home_team_goal_differential')

    print("SVM model accuracy is:: " + str(svm_accuracy))



run_analysis()