import requests

def get_weather_from_coords(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        current = data.get("current_weather", {})
        weather_code = current.get("weathercode", 0)
        temp = current.get("temperature", 0)
        return {
            "weather": weather_code_to_text(weather_code), 
            "temperature": temp 
        }
    except:
        return {"weather": None, "temperature": None}

def weather_code_to_text(code):
    mapping = {
        0: "Clear",
        1: "Mainly Clear", 
        2: "Partly Cloudy",
        3: "Overcast",
        45: "Fog",
        48: "Depositing Rime Fog",
        51: "Drizzle Light",
        53: "Drizzle Moderate",
        55: "Drizzle Dense",
        61: "Rain Slight",
        63: "Rain Moderate",
        65: "Rain Heavy",
        71: "Snow Slight",
        73: "Snow Moderate",
        75: "Snow Heavy",
        95: "Thunderstorm",
    }
    return mapping.get(code, "Clear")