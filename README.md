# -Time-Series-Analysis-Study-Notes
A complete beginner-friendly guide to understanding and applying Time Series Analysis using Python.


# 📈 Time Series Analysis — Study Notes

A complete beginner-friendly guide to understanding and applying Time Series Analysis using Python.

---

## 📌 Table of Contents
- [What is Time Series?](#what-is-time-series)
- [Libraries Used](#libraries-used)
- [Loading and Inspecting the Data](#loading-and-inspecting-the-data)
- [Converting Month to Datetime](#converting-month-to-datetime)
- [Visualizing the Data](#visualizing-the-data)
- [Stationary vs Non-Stationary](#stationary-vs-non-stationary)
- [Tests for Stationarity](#tests-for-stationarity)
- [How to Make Data Stationary](#how-to-make-data-stationary)
- [Understanding ARIMA](#understanding-arima)
- [ACF and PACF](#acf-and-pacf)
- [Train and Test Split](#train-and-test-split)
- [ARIMA Parameters](#arima-parameters)
- [Why ARIMA Failed — Introducing SARIMA](#why-arima-failed--introducing-sarima)

---

## What is Time Series?

A **time series** is data that is collected in a specific order over time. The key difference from regular data is that **the order matters**.

### Examples:
- Your weight measured every morning
- The number of customers visiting a store each day
- Stock price at the end of each day
- Number of steps you walk each hour

### Why Time Series is Important:
You can spot patterns. For example, you might notice that a store always gets more customers on weekends. Once you see these patterns, you can make predictions — *"next Saturday will probably be busy too."*

### Basic Components:

| Component | Description | Example |
|---|---|---|
| **Trend** | Is it generally going up or down? | Sales increasing each month |
| **Seasonality** | Does it repeat regularly? | More ice cream sales during festive season |
| **Noise** | Random ups and downs with no clear pattern | Unexplained daily fluctuations |

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
```

> `warnings.filterwarnings("ignore")` suppresses unnecessary warning messages to keep the output clean.

---

## Loading and Inspecting the Data

The dataset used is the classic **Airline Passengers dataset** — tracking monthly airline passenger counts from 1949 to 1960.

```python
df = pd.read_csv("airline_passengers.csv")
df.head(10)
df.info()
```

### What `df.info()` tells you:

| Column | Non-Null Count | Dtype | Issue |
|---|---|---|---|
| Month | 144 non-null | object | ⚠️ Should be datetime |
| #Passengers | 144 non-null | int64 | ✅ Correct |

> **Key observation:** The `Month` column is stored as `object` (text). For time series analysis it must be converted to `datetime`.

---

## Converting Month to Datetime

### What is DateTime?
DateTime is a data type used to store both date and time together — day, month, year, hour, minute, and second. It represents a specific moment in time.

### Why is Datetime Important in Time Series?

| Reason | Explanation |
|---|---|
| **Plots make sense** | If dates are stored as text, Python won't know they are sequential |
| **Resampling** | Allows grouping by time intervals e.g. average per year |
| **Time-based indexing** | You can easily select data by date range |
| **Forecasting** | Time series models need real sequential dates to predict future points |

```python
# Step 1: Convert Month column to datetime
df['Month'] = pd.to_datetime(df['Month'])

# Step 2: Set Month as the index
df.set_index('Month', inplace=True)

# Confirm the change
df.info()
```

### Before and After:

| | Before | After |
|---|---|---|
| **Index** | RangeIndex (0, 1, 2...) | DatetimeIndex (1949-01-01...) |
| **Month** | Regular column | Index of the dataframe |
| **Data type** | Text (object) | Datetime (datetime64) |

> `inplace=True` modifies the original dataframe directly without needing to reassign it.

---

## Visualizing the Data

```python
df['#Passengers'].plot(title="Monthly Airline Passengers", figsize=(10,5))
plt.xlabel("Date")
plt.ylabel("Number of Passengers")
plt.show()
```

### What the graph shows:

| Component | What you see |
|---|---|
| **Trend** | Clear upward growth from ~100 (1949) to ~600 (1960) |
| **Seasonality** | Regular repeating peaks every year (likely summer) |
| **Growing variance** | Spikes get bigger over time (multiplicative seasonality) |

---

## Stationary vs Non-Stationary

### Stationary Time Series
A stationary time series is one where the data stays relatively stable over time. The pattern does not change.

**Characteristics:**
- No strong upward or downward trend
- No seasonality (repeating patterns)

### Non-Stationary Time Series
A non-stationary time series is one where the data changes over time. The pattern is not stable.

**Characteristics:**
- Has a clear trend (going up or down)
- Has seasonality (repeating patterns)

### Why This Matters:
Most forecasting methods (like ARIMA) require **stationary data** to work properly. If your data is non-stationary, you need to transform it first — remove the trend or seasonality — before using those methods.

| | Stationary | Non-Stationary |
|---|---|---|
| **Trend** | ❌ None | ✅ Has trend |
| **Seasonality** | ❌ None | ✅ Has seasonality |
| **Analogy** | Calm flat sea 🌊 | Stormy sea with rising waves 🌊📈 |

> The airline passenger dataset is **non-stationary** — it has both a clear upward trend and strong seasonality.

---

## Tests for Stationarity

### Augmented Dickey-Fuller (ADF) Test
This is the most common test for checking stationarity.

**How it works:**
- Checks if the time series has a unit root (a mathematical way of saying "non-stationary")
- Gives you a p-value (a number between 0 and 1)

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['#Passengers'])
print(result)
```

### Interpretation:

| p-value | Meaning | Decision |
|---|---|---|
| **< 0.05** | Data is stationary | Reject null hypothesis |
| **≥ 0.05** | Data is non-stationary | Fail to reject null hypothesis |

> The **null hypothesis** of the ADF test assumes the data is non-stationary. A p-value < 0.05 gives enough evidence to reject that assumption.

---

## How to Make Data Stationary

If the time series is non-stationary, we transform the data so its behavior becomes stable over time.

### Common Methods:

| Method | What it removes | Code |
|---|---|---|
| **Differencing** | Removes trend | `df['Diff'] = df['#Passengers'].diff()` |
| **Detrending** | Removes trend | `df['Detrended'] = df['#Passengers'] / df['Trend']` |
| **Log Transformation** | Stabilizes variance | `df['Log'] = np.log(df['#Passengers'])` |

```python
# Differencing
df['Diff_1'] = df['#Passengers'].diff()

# Detrending
df['Trend'] = df['#Passengers'].rolling(window=12, center=True).mean()
df['Detrended_Ratio'] = df['#Passengers'] / df['Trend']

# Log Transformation
df['Log_Passengers'] = np.log(df['#Passengers'])
```

> For the airline dataset, the recommended approach is **log transformation first**, then **differencing** — this handles both the growing variance and the trend.

---

## Understanding ARIMA

**ARIMA = AutoRegressive Integrated Moving Average**

ARIMA is a time series forecasting method that learns from the past to predict the future.

| Letter | Name | Meaning |
|---|---|---|
| **AR** | AutoRegressive | Uses past values to predict future values |
| **I** | Integrated | Differences the data to remove trend and make it stationary |
| **MA** | Moving Average | Learns from past prediction errors to correct future predictions |

> **Note:** The "I" in ARIMA only removes **trend** through differencing. It does **not** remove seasonality. Seasonality requires SARIMA.

### In one sentence:
> ARIMA forecasts future values by **remembering past values (AR)**, working on **cleaned stable data (I)**, and **correcting past mistakes (MA)**.

---

## ACF and PACF

Before fitting ARIMA, we need to decide how many past values and past errors to include. ACF and PACF are the tools that help us do this.

| Tool | Full Name | What it shows | Decides |
|---|---|---|---|
| **ACF** | Autocorrelation Function | How today's value relates to all past values | **q** |
| **PACF** | Partial Autocorrelation Function | How today's value relates to a specific past value, ignoring values in between | **p** |

```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plot_acf(df['#Passengers'], lags=40)
plot_pacf(df['#Passengers'], lags=40)
plt.show()
```

---

## Train and Test Split

### How splitting is different from regular ML:

| | Regular ML | Time Series |
|---|---|---|
| **Splitting method** | Random shuffle is fine | Must split by time only |
| **Order** | Doesn't matter | Order is everything |
| **Data leakage risk** | Lower | Higher if done wrong |

```python
# Define split point (80% train, 20% test)
train_size = int(len(df) * 0.8)

# Split by position, not randomly
train = df.iloc[:train_size]
test = df.iloc[train_size:]

print(f"Training size: {len(train)}")
print(f"Testing size: {len(test)}")

# Verify the split
train.tail()   # where training ends
test.head()    # where testing begins
```

> **Golden rule:** Never let your model see the future while training — that is called **data leakage** and it gives unrealistically good results.

---

## ARIMA Parameters

ARIMA is controlled by 3 parameters written as **ARIMA(p, d, q)**:

| Parameter | Name | Question it answers | Decided by |
|---|---|---|---|
| **p** | AR term | How many past values to use? | PACF |
| **d** | Integrated term | How many times to difference the data? | ADF test |
| **q** | MA term | How many past errors to use? | ACF |

### Examples:

| Parameter | Value | Meaning |
|---|---|---|
| p = 1 | AR(1) | Today depends on yesterday |
| p = 2 | AR(2) | Today depends on last two days |
| d = 0 | No differencing | Data is already stationary |
| d = 1 | Difference once | Subtract previous value once |
| q = 1 | MA(1) | Model uses yesterday's error |

```python
from statsmodels.tsa.arima.model import ARIMA

model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())
```

### Common ARIMA orders:

| Order | Meaning |
|---|---|
| `(1, 1, 1)` | Simple, most commonly used starting point |
| `(0, 1, 0)` | Pure random walk — just differencing |
| `(2, 1, 2)` | More complex — looks further back |

---

## Why ARIMA Failed — Introducing SARIMA

ARIMA failed not because it is wrong, but because it does not understand seasonal patterns.

- ARIMA has **short-term memory**
- It remembers yesterday, but it **forgets last year**

### SARIMA:
SARIMA adds seasonality to ARIMA.

```
ARIMA + Seasonality = SARIMA
```

SARIMA is written as: **ARIMA(p, d, q)(P, D, Q, s)**

| Parameter | Meaning |
|---|---|
| **p, d, q** | Same as regular ARIMA (short term patterns) |
| **P** | Seasonal AR — how many past seasons to look at |
| **D** | Seasonal differencing — removes seasonal trend |
| **Q** | Seasonal MA — corrects past seasonal errors |
| **s** | Length of one seasonal cycle (12 = monthly data with yearly seasonality) |

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
model_fit = model.fit()
print(model_fit.summary())
```

### ARIMA vs SARIMA:

| | ARIMA | SARIMA |
|---|---|---|
| **Seasonality** | ❌ No | ✅ Yes |
| **When to use** | Data with no seasonal pattern | Data with seasonal pattern |
| **Example** | Daily stock prices | Monthly airline passengers |
| **Memory** | Remembers recent past only | Remembers recent past AND same time last year |

> For the airline passengers dataset, **SARIMA is the correct choice** because the data has clear yearly seasonality (s=12).

---

## 📚 Key Takeaways

- Time series data must always **respect time order**
- Always **convert date columns to datetime** and set as index
- Check for **stationarity** using the ADF test before modeling
- Use **differencing, detrending, or log transformation** to make data stationary
- Use **ACF and PACF** to decide ARIMA parameters p and q
- Always split data **by time**, never randomly
- Use **ARIMA** for data without seasonality
- Use **SARIMA** for data with seasonality

---

*Dataset used: Classic Airline Passengers Dataset (1949–1960)*
