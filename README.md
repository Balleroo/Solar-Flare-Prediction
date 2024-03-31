# Solar-Flare-Prediction README

**Version 1.1.0**

**Include a readme file that explains your file structure, how to run your code, and name all third-party modules you are using.**


## Outline
1. Solar Flare notebook
2. Our Modules
3. Third-party Modules
4. Authors


## Solar Flare.ipynb

This notebook shows all the visualizations for our presentation and includes the machine learning models we use for the prediction of energy, duration, x_pos, and y_pos.


## Modules

#### preprocessing.py

- function parse_date(sdatex, stimex)

This is used to parse the strings from the date time data in the dataframe.
This allows us to create a more robust datetime format for its start, peak, and end.

- function preprocess(fname)

This allows us to read in the csv file containing the whole solar flare data that we need to use for the project.

1. We drop certain columns and values that are no use to our project goal and helps to save time when processing the dataframe.
2. We also make certain changes to the datetime formats given so that they become easier to handle in the machine learning section.
3. Along with the datetime, we changes values of the energy which is our main focus. Since they are in a range format, using machine learning makes it extremely complicated, therefore we converted them into numbers from 0 to 8 and will reconvert after the machine learning process
4. We also drop certain unwanted columns at the end and change the duration column to a logarithmic scale since it is slightly skewed.

Example: pd_df = preprocess("hessi.solar.flare.UP_To_2018.csv")

- function preprocess_plots(fname)

This allows us to read in the csv file containing the whole solar flare data and process the data for the plots

Example: pd_df = preprocess("hessi.solar.flare.UP_To_2018.csv")

#### trend_plots.py

- function plots(df)
- 
Takes in a pandas DataFrame and plots the data.

Example: plots(df)

#### ML_prediction.py

- function train_test_spilt(df)

Takes in a pandas DataFrame, splits it into train/test set and retunrs.

Example: X_train, X_test = train_test_spilt(df)

- function duration_prediction(cols = [])

Takes in the columns name using to predict the duration, returns the regressor, confidence score, and MSE.

Example: model, score, MSE = duration_prediction(['total_counts', 'energy_kev', 'x_pos', 'y_pos'])

- function energy_prediction(cols = [])

Takes in the columns name using to predict the energy_kev, returns the classifier and accuracy score.

Example: model, acc = energy_prediction(['duration', 'x_pos', 'y_pos'])

- function x_y_pos_prediction(cols = [])

Takes in the columns name using to predict the x_pos and y_pos, returns the regressor, confidence score, and MSE.

Example: model_x, score_x, MSE_x, model_y, score_y, MSE_y = x_y_pos_prediction(['duration', 'total_counts', 'energy_kev'])


#### Add your python files...

- function ...

- function ...


#### More python files..

- fucntion ...

- function ...

## Third-party Modules
numpy, pandas, matplotlib, seaborn, scikit-learn, scipy, datetime


## Authors
Group 5:
Haaris Rahman, Kevin Mills, Shaan Bhalaru, Yongxing Chen, Shusen Lin



