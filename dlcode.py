
!pip -q install tensorflow pandas numpy matplotlib scikit-learn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab import files
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import io

np.random.seed(42)
tf.random.set_seed(42)

print("Upload train and test CSV files")
uploaded = files.upload()
dataframes = {}

for name, content in uploaded.items():
    df = pd.read_csv(io.BytesIO(content))
    df.columns = df.columns.str.strip().str.lower()
    if "date" in df.columns and "meantemp" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        dataframes[name.lower()] = df

train_df = next((v for k, v in dataframes.items() if "train" in k), None)
test_df = next((v for k, v in dataframes.items() if "test" in k), None)

if train_df is None:
    if not dataframes:
        raise ValueError("Upload a valid Daily Delhi Climate CSV file.")
    df = next(iter(dataframes.values()))
    df = df.sort_values("date").reset_index(drop=True)
    split = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

features = ["meantemp", "humidity", "wind_speed", "meanpressure"]

for frame in [train_df, test_df]:
    missing = set(features) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    for col in features:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame.dropna(subset=["date"] + features, inplace=True)
    frame.drop_duplicates("date", keep="last", inplace=True)
    frame.sort_values("date", inplace=True)
    frame.reset_index(drop=True, inplace=True)

scaler = MinMaxScaler()
scaler.fit(train_df[features])
train_scaled = scaler.transform(train_df[features])
test_scaled = scaler.transform(test_df[features])

LOOKBACK = 30
temp_idx = 0

def make_sequences(data):
    X, y = [], []
    for i in range(LOOKBACK, len(data)):
        X.append(data[i-LOOKBACK:i])
        y.append(data[i, temp_idx])
    return np.array(X), np.array(y)

X_train, y_train = make_sequences(train_scaled)

if len(X_train) < 2:
    raise ValueError("Not enough training data. Upload a larger training CSV.")

model = Sequential([
    tf.keras.Input(shape=(LOOKBACK, len(features))),
    LSTM(64, return_sequences=True),
    Dropout(0.2),
    LSTM(32),
    Dense(16, activation="relu"),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse", metrics=["mae"])

model.fit(
    X_train, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    callbacks=[EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )],
    verbose=1
)

combined = np.vstack([train_scaled, test_scaled])
X_test, y_test = [], []

for i in range(len(train_scaled), len(combined)):
    if i >= LOOKBACK:
        X_test.append(combined[i-LOOKBACK:i])
        y_test.append(combined[i, temp_idx])

X_test, y_test = np.array(X_test), np.array(y_test)

def inverse_temp(values):
    values = np.asarray(values).reshape(-1)
    full = np.zeros((len(values), len(features)))
    full[:, temp_idx] = values
    return scaler.inverse_transform(full)[:, temp_idx]

if len(X_test):
    predictions = inverse_temp(model.predict(X_test, verbose=0))
    actual = inverse_temp(y_test)
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))

    results = pd.DataFrame({
        "Date": test_df["date"].iloc[:len(actual)].to_numpy(),
        "Actual Temperature": actual,
        "Predicted Temperature": predictions
    })

    print(f"\nMAE: {mae:.3f} °C")
    print(f"RMSE: {rmse:.3f} °C")
    display(results.head(10))

    plt.figure(figsize=(12, 5))
    plt.plot(results["Date"], results["Actual Temperature"], label="Actual")
    plt.plot(results["Date"], results["Predicted Temperature"], label="Predicted")
    plt.xlabel("Date")
    plt.ylabel("Mean Temperature (°C)")
    plt.title("Actual vs Predicted Temperature")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    results.to_csv("weather_predictions.csv", index=False)
else:
    print("Not enough test data to evaluate.")

# Forecast the next 7 days using the latest known weather features
all_data = pd.concat([train_df, test_df]).sort_values("date")
all_data = all_data.drop_duplicates("date", keep="last")
all_scaled = scaler.transform(all_data[features])
last_data = all_scaled[-LOOKBACK:].copy()

if len(last_data) == LOOKBACK:
    future_temps = []

    for _ in range(7):
        x = last_data[-LOOKBACK:].reshape(1, LOOKBACK, len(features))
        next_temp = model.predict(x, verbose=0)[0, 0]
        future_temps.append(next_temp)

        next_row = last_data[-1].copy()
        next_row[temp_idx] = next_temp
        last_data = np.vstack([last_data, next_row])

    forecast = pd.DataFrame({
        "Date": pd.date_range(
            all_data["date"].max() + pd.Timedelta(days=1), periods=7
        ),
        "Forecast Temperature (°C)": inverse_temp(future_temps)
    })

    print("\n7-Day Temperature Forecast")
    display(forecast)

    plt.figure(figsize=(9, 4))
    plt.plot(forecast["Date"], forecast["Forecast Temperature (°C)"],
             marker="o")
    plt.title("7-Day Temperature Forecast")
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    forecast.to_csv("weather_forecast_7_days.csv", index=False)

model.save("weather_lstm_model.keras")
print("\nModel saved as weather_lstm_model.keras")
print("Prediction and forecast CSV files saved.")
