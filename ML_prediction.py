import pandas as pd
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_test_spilt(df):
    """
    Takes in a pandas DataFrame, 
    splits it into train/test set and retunrs

    :param df: input DataFrame
    :return: two DataFrame of train/test set
    """
    assert isinstance(df, pd.DataFrame)
    X_train, X_test = train_test_split(df, test_size=0.2)
    return X_train, X_test


def duration_prediction(cols = []):
    """
    Takes in the columns name using to predict the duration,
    returns the regressor, confidence score, and MSE

    :param cols: columns used to predict duration
    :return: duration regressor, confidence score, and MSE
    """
    assert len(cols) > 0

    fname = "hessi.solar.flare.UP_To_2018.csv"
    pd_df = preprocess(fname)

    X_train, X_test = train_test_spilt(pd_df)

    y_train_duration = X_train['duration']
    X_train_duration = X_train[cols]
    y_test_duration = X_test['duration']
    X_test_duration = X_test[cols]

    # Random Forest
    from sklearn.ensemble import RandomForestRegressor
    random_forest_regressor = RandomForestRegressor(n_jobs=-1).fit(X_train_duration, y_train_duration)
    random_forest_predictions = random_forest_regressor.predict(X_test_duration)
    random_forest_predictions_train = random_forest_regressor.predict(X_train_duration)
    random_forest_MSE_train = mean_squared_error(y_train_duration, random_forest_predictions_train)
    random_forest_MSE = mean_squared_error(y_test_duration, random_forest_predictions)
    random_forest_score = random_forest_regressor.score(X_test_duration, y_test_duration)
    random_forest_score_train = random_forest_regressor.score(X_train_duration, y_train_duration)
    
    return random_forest_regressor, (random_forest_score, random_forest_score_train), (random_forest_MSE, random_forest_MSE_train)


def energy_prediction(cols = []):
    """
    Takes in the columns name using to predict the energy_kev,
    returns the classifier and accuracy score

    :param cols: columns used to predict energy_kev
    :return: energy_kev classifer and its accuracy score
    """
    assert len(cols) > 0

    fname = "hessi.solar.flare.UP_To_2018.csv"
    pd_df = preprocess(fname)

    X_train, X_test = train_test_spilt(pd_df)

    y_train_energy = X_train['energy_kev']
    X_train_energy = X_train[cols]
    y_test_energy = X_test['energy_kev']
    X_test_energy = X_test[cols]

    # Random Forest
    from sklearn.ensemble import RandomForestClassifier
    random_forest_classifier = RandomForestClassifier(n_jobs=-1).fit(X_train_energy, y_train_energy)
    random_forest_score = random_forest_classifier.score(X_test_energy, y_test_energy)
    random_forest_score_train = random_forest_classifier.score(X_train_energy, y_train_energy)
    
    return random_forest_classifier, (random_forest_score, random_forest_score_train)

def x_y_pos_prediction(cols = []):
    """
    Takes in the columns name using to predict the x_pos and y_pos,
    returns the regressor, confidence score, and MSE

    :param cols: columns used to predict duration
    :return: x_pos regressor, confidence score, and MSE
                y_pos regressor, confidence score, and MSE
    """
    assert len(cols) > 0

    fname = "hessi.solar.flare.UP_To_2018.csv"
    pd_df = preprocess(fname)

    X_train, X_test = train_test_spilt(pd_df)

    y_train_xpos = X_train['x_pos']
    X_train_xpos = X_train[cols]
    y_test_xpos = X_test['x_pos']
    X_test_xpos = X_test[cols]

    # Random Forest
    from sklearn.ensemble import RandomForestRegressor
    random_forest_regressor_x = RandomForestRegressor(n_jobs=-1).fit(X_train_xpos, y_train_xpos)
    random_forest_predictions_x = random_forest_regressor_x.predict(X_test_xpos)
    random_forest_predictions_train_x = random_forest_regressor_x.predict(X_train_xpos)
    random_forest_MSE_train_x = mean_squared_error(y_train_xpos, random_forest_predictions_train_x)
    random_forest_MSE_x = mean_squared_error(y_test_xpos, random_forest_predictions_x)
    random_forest_score_x = random_forest_regressor_x.score(X_test_xpos, y_test_xpos)
    random_forest_score_train_x = random_forest_regressor_x.score(X_train_xpos, y_train_xpos)

    y_train_ypos = X_train['y_pos']
    X_train_ypos = X_train[cols]
    y_test_ypos = X_test['y_pos']
    X_test_ypos = X_test[cols]

    # Random Forest
    random_forest_regressor_y = RandomForestRegressor(n_jobs=-1).fit(X_train_ypos, y_train_ypos)
    random_forest_predictions_y = random_forest_regressor_y.predict(X_test_ypos)
    random_forest_predictions_train_y = random_forest_regressor_x.predict(X_train_ypos)
    random_forest_MSE_train_y = mean_squared_error(y_train_ypos, random_forest_predictions_train_y)
    random_forest_MSE_y = mean_squared_error(y_test_ypos, random_forest_predictions_y)
    random_forest_score_y = random_forest_regressor_y.score(X_test_ypos, y_test_ypos)
    random_forest_score_train_y = random_forest_regressor_y.score(X_train_ypos, y_train_ypos)

    return random_forest_regressor_x, (random_forest_score_x, random_forest_score_train_x), \
            (random_forest_MSE_x, random_forest_MSE_train_x), random_forest_regressor_y, \
            (random_forest_score_y, random_forest_score_train_y), (random_forest_MSE_y, random_forest_MSE_train_y)