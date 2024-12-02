from json import JSONDecoder

import requests
import json


url = "https://archive-api.open-meteo.com/v1/archive"

def parse_json_data(json_data):
    return 0


def retrieve_weather_data(latitude: float, longitude: float, start_date: str, end_date: str):

    params = {"latitude": latitude, "longitude": longitude, "start_date": start_date, "end_date": end_date,
              "hourly": "surface_pressure", "daily": "apparent_temperature_mean,rain_sum"}

    json_response = requests.get(url, params=params).json()

    mean_pressure = calculate_mean_pressure(json_response["hourly"]["time"],json_response["hourly"]["surface_pressure"])

    data = {"mean_pressure": mean_pressure,
            "temperature": json_response["daily"]["apparent_temperature_mean"],
            "was_raining": [True if rain > 0 else False for rain in json_response["daily"]["rain_sum"]]}

    return data


def calculate_mean_pressure(time, pressure):
    time_pressure = list(zip(time, pressure))

    pressure_daily = []
    prev_date = time_pressure[0][0].split("T")[0].strip()
    pressure_sum = 0
    counter = 0
    for time,pressure in time_pressure:
        curr_date = time.split("T")[0].strip()
        if curr_date == prev_date:
            pressure_sum += pressure
            counter += 1
        else:
            pressure_daily.append(round(pressure_sum / counter))
            prev_date = curr_date
            pressure_sum = pressure
            counter = 0

    return pressure_daily