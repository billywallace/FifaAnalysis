import matplotlib.pyplot as plt
import numpy as np


# Calculate the range of fifa overall values
def plot_linear_regression_space(df_predictions):
    max_value = df_predictions['home_fifa_overall'].max() + 3
    min_vale = df_predictions['home_fifa_overall'].min() - 3

    x = np.linspace(min_vale, max_value, 100)
    y = np.linspace(min_vale, max_value, 100)
    x, y = np.meshgrid(x, y)

    z = x * 0.03934139 - y * 0.03971793 + 1.473610850817787

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot the data inputs as a 3d scatter plot
    xs = np.array(df_predictions['home_fifa_overall'])
    ys = np.array(df_predictions['away_fifa_overall'])
    zs = np.array(df_predictions['home_team_goal_differential'])
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('Home Team Fifa Rating')
    ax.set_ylabel('Away Team Fifa Rating')
    ax.set_zlabel('Home Team Score Differential')

    # Plot prediction plane from linear regression
    ax.plot_surface(x, y, z, label='Prediction Plane')
    #ax.legend()


    plt.show()