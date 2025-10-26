import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import sys

# =========================
# SETTINGS
# =========================
latitude = 21.8022
longitude = 81.8463
start_date = "20220101"
end_date = "20221231"
params_list = ["T2M", "T2M_MAX", "T2M_MIN", "WS10M", "WS10M_MAX", "WS10M_MIN", "PRECTOT"]
community = "AG"  # use AG or SB; avoid unknown values like RE

# =========================
# BUILD REQUEST
# =========================
base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
params = {
    "parameters": ",".join(params_list),
    "community": community,
    "longitude": longitude,
    "latitude": latitude,
    "start": start_date,
    "end": end_date,
    "format": "JSON",
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

print("Downloading weather data...")
try:
    resp = requests.get(base_url, params=params, headers=headers, timeout=30)
except requests.RequestException as e:
    print("Network error while trying to download data:", e)
    sys.exit(1)

if resp.status_code == 200:
    try:
        data_json = resp.json()
    except ValueError:
        print("Error: response is not valid JSON. Response text:")
        print(resp.text[:1000])
        sys.exit(1)
else:
    print(f"Error fetching data: {resp.status_code} {resp.reason}")
    print("Server response (first 1000 chars):")
    print(resp.text[:1000])
    # Helpful fallback: show final request URL for debugging
    print("Final request URL:", resp.url)
    sys.exit(1)

# =========================
# EXTRACT PARAMETERS
# =========================
if "properties" not in data_json or "parameter" not in data_json["properties"]:
    print("Unexpected JSON structure. Full response preview (first 1000 chars):")
    print(str(data_json)[:1000])
    sys.exit(1)

daily_data = data_json["properties"]["parameter"]

# choose precipitation key if available
precip_key = None
for k in ("PRECTOTCORR", "PRECTOT", "PRECIP"):
    if k in daily_data:
        precip_key = k
        break

required_keys = ["T2M", "T2M_MAX", "T2M_MIN", "WS10M"]
for rk in required_keys:
    if rk not in daily_data:
        print(f"Required key missing from response: {rk}")
        print("Available keys:", list(daily_data.keys()))
        sys.exit(1)

dates = sorted(daily_data["T2M"].keys())

# =========================
# BUILD DATAFRAME
# =========================
df = pd.DataFrame({
    "DATE": pd.to_datetime(dates, format="%Y%m%d"),
    "T2M": [daily_data["T2M"][d] for d in dates],
    "T2M_MAX": [daily_data["T2M_MAX"][d] for d in dates],
    "T2M_MIN": [daily_data["T2M_MIN"][d] for d in dates],
    "WS10M": [daily_data["WS10M"].get(d, np.nan) if isinstance(daily_data["WS10M"], dict) else np.nan for d in dates],
})

if precip_key:
    df[precip_key] = [daily_data[precip_key].get(d, np.nan) for d in dates]

df = df.sort_values("DATE").reset_index(drop=True)
df = df.dropna(subset=["T2M"])

# =========================
# PLOT HISTORICAL TEMPERATURE
# =========================
plt.figure(figsize=(10,5))
plt.plot(df["DATE"], df["T2M"], label="Daily Temp", color="tab:blue")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Historical Daily Temperature")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# LINEAR REGRESSION PREDICTION
# =========================
df["Days"] = (df["DATE"] - df["DATE"].min()).dt.days
X = df[["Days"]].values
y = df["T2M"].values

model = LinearRegression()
model.fit(X, y)

last_day = int(df["Days"].max())
future_days = np.arange(last_day + 1, last_day + 366).reshape(-1, 1)
future_preds = model.predict(future_days)
future_dates = pd.date_range(df["DATE"].max() + pd.Timedelta(days=1), periods=365)

plt.figure(figsize=(10,5))
plt.plot(df["DATE"], y, label="Past Temp", color="tab:blue")
plt.plot(future_dates, future_preds, label="Predicted Temp", color="tab:red")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.title("Future Temperature Prediction (Next 365 Days)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =========================
# SAVE PREDICTIONS
# =========================
pred_df = pd.DataFrame({"Date": future_dates, "Predicted_Temp": future_preds})
pred_df.to_csv("predicted_temperature.csv", index=False)
print("Saved predicted_temperature.csv")
print("Precip key used:", precip_key)