import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
    
def transform_data(df):

    # Renaming the columns
    df.rename(
        columns = {
            "Date": "date",
            "Rented Bike Count": "rented_bike_count",
            "Hour": "hour",
            "Temperature(°C)": "temperature",
            "Humidity(%)": "humidity",
            "Wind speed (m/s)": "wind_speed",
            "Visibility (10m)": "visibility",
            "Dew point temperature(°C)": "dew_point_temperature",
            "Solar Radiation (MJ/m2)": "solar_radiation",
            "Rainfall(mm)": "rainfall",
            "Snowfall (cm)": "snowfall",
            "Seasons": "season",
            "Holiday": "is_holiday",
            "Functioning Day": "functioning_day",
        },
        inplace = True
    )

    # Extracting temporal dimensions from the date
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"] > 4
    df["is_weekend"] = df["is_weekend"].astype(int)

    # Moving the columns to front of the data set
    colnames = ["year", "month", "day", "day_of_week", "is_weekend"]
    for i in range(len(colnames)):
        column = df.pop(colnames[i])
        df.insert(i + 1, colnames[i], column)
    
    # One-hot encoding seasons
    df = pd.get_dummies(df, columns=["season"])
    dummy_columns = ["season_Autumn", "season_Spring", "season_Summer", "season_Winter"]
    for dummy in dummy_columns:
        df[dummy] = df[dummy].astype(int)

    # Converting dummy variables to binary
    df["functioning_day"] = df["functioning_day"].map({"No": 0, "Yes": 1})
    df["is_holiday"] = df["is_holiday"].map({"No Holiday": 0, "Holiday": 1})

    # Dropping non-functioning days
    df = df[df["functioning_day"] == 1]

    # Dropping the functioning day and dew point temperature variables
    df.drop(columns = ["functioning_day", "dew_point_temperature"], inplace = True)

    # Log transforming the rented bike count (target) column to adjust for skew
    df["rented_bike_count"] = np.log1p(df["rented_bike_count"])

    # Creating demand groups
    peak_hours = [18, 19, 17, 20, 21, 8, 16, 22]
    normal_hours = [15, 14, 13, 12, 23, 9, 7, 11]
    low_hours = [0, 10, 1, 2, 6, 3, 5, 4]
    df["is_peak"] = df["hour"].isin(peak_hours)
    df["is_normal"] = df["hour"].isin(normal_hours)
    df["is_low"] = df["hour"].isin(low_hours)

    # Separating the feature matrix and target vector
    X = np.array(df.drop(columns = ["rented_bike_count"]))
    Y = np.array(df["rented_bike_count"])

    # Splitting the data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 42, shuffle = False)

    return X_train, X_test, Y_train, Y_test


sbd = pd.read_csv("SeoulBikeData.csv", encoding = "latin-1")

X_train, X_test, Y_train, Y_test = transform_data(sbd)