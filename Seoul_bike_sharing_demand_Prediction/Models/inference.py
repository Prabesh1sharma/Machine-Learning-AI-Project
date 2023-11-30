import pickle
import os
import pandas as pd 
import numpy as np 
from datetime import datetime


class Inference:
    def __init__(self, model_path, sc_path):
        self.model_path = model_path
        self.sc_path = sc_path

        if os.path.exists(self.model_path) and os.path.exists(self.sc_path):
            self.model = pickle.load(open(self.model_path, "rb"))
            self.sc = pickle.load(open(self.sc_path, "rb"))
        else:
            print("Model or Standar Scaler Path is no correct!!")

    def get_string_to_datetime(self, date):
        dt = datetime.strptime(date, "%d/%m/%Y")
        return {"day":dt.day, "month":dt.month, "year":dt.year, "week_day":dt.strftime("%A")}
    

    def season_to_df(self, seasons):
        seasons_col = ['Spring', 'Summer', 'Winter']
        seasons_data = np.zeros((1, len(seasons_col)), dtype="int")

        df_seasons = pd.DataFrame(seasons_data, columns=seasons_col)
        if seasons in seasons_col:
            df_seasons[seasons]=1
        return df_seasons
    
    def days_df(self, week_day):
        days_names = ['Monday','Saturday','Sunday','Thursday','Tuesday','Wednesday']
        days_name_data = np.zeros((1, len(days_names)),dtype="int")

        df_days = pd.DataFrame(days_name_data, columns=days_names)

        if week_day in days_names:
            df_days[week_day]=1

        return df_days
    

    def users_input(self,):
        print("Enter correct information to predict Rented Bike count for a day with respect to time")
        date = input("Date (dd/mm/yyyy): ")
        hour = int(input("Hours (0-23) : "))
        temperature = float(input("Temperature in C :"))
        humidity=float(input("Humidity :"))
        wind_speed = float(input("Wind Speed :"))
        visibility =float(input("Visibility :"))
        solar_radiation = float(input("Solar Radiation :"))
        rainfall = float(input("Rainfall :"))
        snowfall = float(input("Snowfall :"))
        seasons = input("Seasons (Spring / Summer / Winter / Autumn): ")
        holiday = input("Holiday (Holiday / No Holiday): ")
        functioning_day = input("Functioning Day (Yes/No): ")

        Holiday_dic = {"No Holiday":0, "Holiday":1}
        finctioning_day = {"No":0, "Yes":1}
        str_to_date = self.get_string_to_datetime(date)
        u_input_list = [hour, temperature, humidity, wind_speed, visibility, solar_radiation, rainfall, snowfall,
               Holiday_dic[holiday], finctioning_day[functioning_day],
               str_to_date["day"], str_to_date["month"], str_to_date["year"]]

        features_name = ['Hour', 'Temperature(°C)', 'Humidity(%)', \
            'Wind speed (m/s)', 'Visibility (10m)', \
            'Solar Radiation (MJ/m2)', 'Rainfall(mm)', 'Snowfall (cm)', \
            'Holiday', 'Functioning Day', 'Day', 'Month', 'Year']

        df_u_input = pd.DataFrame([u_input_list], columns=features_name)
        df_seasons = self.season_to_df(seasons)
        df_days = self.days_df(str_to_date["week_day"])
        df_for_pred = pd.concat([df_u_input, df_seasons,df_days], axis=1)

        return df_for_pred
    

    def prediction(self):
        df = self.users_input()
        scaled_data = self.sc.transform(df)
        prediction = self.model.predict(scaled_data)
        return prediction


if __name__ == "__main__":

    ml_model_path = r"D:\Data Science\ML Project\Seoul_bike_sharing_demand_Prediction\Models\xgboost_regressor_r2_0.949_v1.pkl"
    standard_scaler_path = r"D:\Data Science\ML Project\Seoul_bike_sharing_demand_Prediction\Models\sc.pkl"
    inference = Inference(ml_model_path, standard_scaler_path)

    pred = inference.prediction()

    print(f"Rented Bike Count prediction with respect to date and time : {round(pred.tolist()[0])}")
