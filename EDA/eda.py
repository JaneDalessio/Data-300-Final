import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
    
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
    df.drop(columns = ["functioning_day", "dew_point_temperature", "date"], inplace = True)

    # Log transforming the rented bike count (target) column to adjust for skew
    df["rented_bike_count"] = np.log1p(df["rented_bike_count"])

     # Adding lag features to enhance prediction outcomes
    lag_values = [1, 2, 3, 24, 48, 168]
    for lag in lag_values:
        df[f"rbc_{lag}h"] = df["rented_bike_count"].shift(lag)
    
    # Creating missing lag value flags
    flags = ["rbc_1h_m", "rbc_2h_m", "rbc_3h_m", "rbc_24h_m", "rbc_48h_m", "rbc_168h_m"]
    lags = ["rbc_1h", "rbc_2h", "rbc_3h", "rbc_24h", "rbc_48h", "rbc_168h"]
    for i in range(len(flags)):
        df[flags[i]] = df[lags[i]].isna().astype(int) 
    
    # Fill missing lag values
    hourly_means = df.groupby(df["hour"])["rented_bike_count"].mean()
    df['rbc_1h'] = df['rbc_1h'].fillna(df["hour"].map(hourly_means))
    df["rbc_2h"] = df["rbc_2h"].fillna(df["rbc_1h"])
    df["rbc_3h"] = df["rbc_3h"].fillna(df["rbc_2h"])
    df["rbc_24h"] = df["rbc_24h"].fillna(df["rbc_3h"])
    df["rbc_48h"] = df["rbc_48h"].fillna(df["rbc_24h"])
    df["rbc_168h"] = df["rbc_168h"].fillna(df["rbc_48h"])

    # Implementing rolling statistics
    shift_value = df["rented_bike_count"].shift(1)
    windows = [3, 6, 12, 24, 48, 168]
    for window in windows:
        df[f"rbc_mean_{window}h"] = shift_value.rolling(window, min_periods = 1).mean()
        if window == 3 or window == 24:
            df[f"rbc_std_{window}h"] = shift_value.rolling(window, min_periods = 1).std()

    # Creating missing rolling statistic flags
    flags = ["rbc_mean_3h_m", "rbc_std_3h_m", "rbc_mean_6h_m", "rbc_mean_12h_m", "rbc_mean_24h_m", "rbc_std_24h_m", "rbc_mean_48h_m", "rbc_mean_168h_m"]
    lags = ["rbc_mean_3h", "rbc_std_3h", "rbc_mean_6h", "rbc_mean_12h", "rbc_mean_24h", "rbc_std_24h", "rbc_mean_48h", "rbc_mean_168h"]
    for i in range(len(flags)):
        df[flags[i]] = df[lags[i]].isna().astype(int) 

    # Fill missing rolling statistics
    hourly_stds = df.groupby(df["hour"])["rented_bike_count"].std()
    df["rbc_mean_3h"] = df["rbc_mean_3h"].fillna(df["hour"].map(hourly_means))
    df["rbc_std_3h"] = df["rbc_std_3h"].fillna(df["hour"].map(hourly_stds))
    df["rbc_mean_6h"] = df["rbc_mean_6h"].fillna(df["rbc_mean_3h"])
    df["rbc_mean_12h"] = df["rbc_mean_12h"].fillna(df["rbc_mean_6h"])
    df["rbc_mean_24h"] = df["rbc_mean_24h"].fillna(df["rbc_mean_12h"])
    df["rbc_std_24h"] = df["rbc_std_24h"].fillna(df["rbc_std_3h"])
    df["rbc_mean_48h"] = df["rbc_mean_48h"].fillna(df["rbc_mean_24h"])
    df["rbc_mean_168h"] = df["rbc_mean_168h"].fillna(df["rbc_mean_48h"])

    # Creating demand groups
    peak_hours = [18, 19, 17, 20, 21, 8, 16, 22]
    normal_hours = [15, 14, 13, 12, 23, 9, 7, 11]
    low_hours = [0, 10, 1, 2, 6, 3, 5, 4]
    df["is_peak"] = df["hour"].isin(peak_hours)
    df["is_normal"] = df["hour"].isin(normal_hours)
    df["is_low"] = df["hour"].isin(low_hours)
    df["is_peak"] = df["is_peak"].astype(int)
    df["is_normal"] = df["is_normal"].astype(int)
    df["is_low"] = df["is_low"].astype(int)

    return df

def split(df):

    # Separating the feature matrix and target vector
    X = np.array(df.drop(columns = ["rented_bike_count"]))
    Y = np.array(df["rented_bike_count"])

    split_index = int(len(X) * 0.75)
    # Splitting the data into train and test sets
    X_train = X[:split_index]
    X_test = X[split_index:]
    Y_train = Y[:split_index]
    Y_test = Y[split_index:]

    return X_train, X_test, Y_train, Y_test

def rbc_kde_plot(df, col = "green"):
    
    # Kernel Density Estimate Graph
    sns.kdeplot(df, x = "Rented Bike Count", color = col)
    plt.title("Kernel Density Estimate for Rented Bike Count")
    plt.show()

def rbc_boxplot(df, col = "green"):

    # Boxplot Graph
    sns.boxplot(df , x = "Rented Bike Count", color = col)
    plt.title("Boxplot for Rented Bike Count")
    plt.show()

def corr_matrix(df):
    
    # Correlation Matrix using Heatmap Graphic
    matrix = df.corr(numeric_only = True)
    sns.heatmap(matrix, cmap = "coolwarm")
    plt.title("Correlation Matrix for Variables in Seoul Bike Sharing Data")
    plt.show()

def demand_bar_plot(df):

    # Creating a grouping variable
    conditions = [
        df["is_peak"],
        df["is_normal"],
        df["is_low"]
    ]
    choices = [1, 2, 3]
    df["demand"] = np.select(conditions, choices, default = 9)

    # Creating the bar plot for Rented Bike Count
    order = df.groupby("hour")["rented_bike_count"].mean().sort_values(ascending = False).index
    sns.barplot(df, x = "rented_bike_count", y = "hour", orient = "h", errorbar = None, order = order, hue = "demand", palette = ['red','orange','yellow'])
    plt.title("Bike Rental Frequency During Each Hour of the Day")
    plt.show()
    
