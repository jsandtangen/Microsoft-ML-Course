# Time Series Forecasting

Dette er en kompakt teknisk README for repetisjon av tidsserier. Fokuset er på typiske imports, plotting, preprocessing, train/test split, ARIMA/SARIMA, SVR, walk-forward validation og evaluering.

---

## 1. Typiske biblioteker

```python id="xnuxfd"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit
```

Installer ved behov:

```bash id="o40ie7"
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels
```

---

## 2. Hva er tidsseriedata?

Tidsseriedata er observasjoner sortert etter tid.

Eksempler:

```text id="lm9x3f"
strømforbruk per time
aksjekurs per dag
temperatur per dag
salg per uke
nettrafikk per minutt
CO2 per måned
```

En tidsserie har ofte:

```text id="lbnx1o"
trend       → langsiktig økning eller nedgang
seasonality → gjentakende mønster
noise       → tilfeldig variasjon
outliers    → ekstreme verdier
level shift → brå endring i nivå
```

---

## 3. Typisk pipeline

```text id="lbk6sw"
Load data
→ parse datetime
→ set datetime index
→ sort by time
→ inspect missing values
→ visualize
→ split train/test by time
→ scale if needed
→ train model
→ forecast
→ inverse transform
→ evaluate
```

Viktig: Ved tidsserier skal man ikke bruke tilfeldig `train_test_split`. Testdata må ligge etter treningsdata i tid.

---

## 4. Laste inn tidsseriedata

```python id="ghyjs0"
df = pd.read_csv("data/energy.csv")

df.head()
df.info()
df.shape
```

Konverter dato:

```python id="kfme6t"
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp")
df = df.set_index("timestamp")
```

Sjekk frekvens:

```python id="f4iqb3"
df.index.min(), df.index.max()
df.index.freq
```

Hvis frekvens mangler:

```python id="bdo0dn"
df = df.asfreq("H")   # hourly
# df = df.asfreq("D") # daily
# df = df.asfreq("M") # monthly
```

---

## 5. Enkel inspeksjon

```python id="dl1p2r"
df.head()
df.tail()
df.describe()
df.isna().sum()
```

Sjekk duplikater i index:

```python id="faod45"
df.index.duplicated().sum()
```

Fjern duplikater:

```python id="p9x4h6"
df = df[~df.index.duplicated(keep="first")]
```

---

## 6. Plot tidsserien

```python id="qzcatx"
df["load"].plot(figsize=(14, 5))
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("Time series")
plt.show()
```

Plot bestemt periode:

```python id="fq6h15"
df.loc["2014-07-01":"2014-07-07", "load"].plot(figsize=(14, 5))
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("Selected period")
plt.show()
```

---

## 7. Manglende verdier

Sjekk:

```python id="qo5zjq"
df.isna().sum()
```

Forward fill:

```python id="fpni98"
df["load"] = df["load"].ffill()
```

Backward fill:

```python id="yj2oaw"
df["load"] = df["load"].bfill()
```

Interpolering:

```python id="ludrtp"
df["load"] = df["load"].interpolate()
```

Rolling mean kan også brukes for glatting:

```python id="zk8ral"
df["load_rolling"] = df["load"].rolling(window=24).mean()
```

---

## 8. Resampling

Resampling brukes for å endre tidsoppløsning.

Fra time til dag:

```python id="zhccai"
daily = df["load"].resample("D").mean()
```

Fra dag til måned:

```python id="a4d5cc"
monthly = df["load"].resample("M").mean()
```

Andre vanlige aggregeringer:

```python id="k56xii"
df["load"].resample("D").sum()
df["load"].resample("D").max()
df["load"].resample("D").min()
```

---

## 9. Rolling statistics

Rolling mean brukes ofte for å se trend.

```python id="bq8ln7"
df["rolling_24h"] = df["load"].rolling(window=24).mean()
df["rolling_7d"] = df["load"].rolling(window=24*7).mean()
```

Plot:

```python id="5ohdp3"
df[["load", "rolling_24h", "rolling_7d"]].plot(figsize=(14, 5))
plt.show()
```

Rolling standard deviation:

```python id="a9b20v"
df["rolling_std"] = df["load"].rolling(window=24).std()
```

---

## 10. Decomposition

Decomposition deler tidsserien i trend, sesong og residualer.

```python id="w1vf4k"
result = seasonal_decompose(
    df["load"],
    model="additive",
    period=24
)

result.plot()
plt.show()
```

For timebasert strømdata kan `period=24` brukes for daglig sesongmønster.

Typiske perioder:

```text id="l66e9f"
hourly data med daglig sesong      → period=24
daily data med ukentlig sesong     → period=7
monthly data med årlig sesong      → period=12
```

---

## 11. Stationarity

Mange klassiske tidsseriemodeller, spesielt ARIMA, fungerer best når serien er stationær.

Stationær betyr at serien har omtrent stabil:

```text id="z6dniz"
mean
variance
autocorrelation
```

ADF-test:

```python id="v0x7ku"
result = adfuller(df["load"].dropna())

print("ADF statistic:", result[0])
print("p-value:", result[1])
```

Tolkning:

```text id="hw9ker"
p-value < 0.05 → sannsynligvis stationær
p-value > 0.05 → sannsynligvis ikke-stationær
```

---

## 12. Differencing

Differencing brukes for å gjøre en ikke-stationær serie mer stationær.

```python id="m7360j"
df["load_diff"] = df["load"].diff()
```

Plot:

```python id="72s3q9"
df["load_diff"].plot(figsize=(14, 5))
plt.show()
```

Seasonal differencing:

```python id="i0y3zn"
df["load_seasonal_diff"] = df["load"].diff(24)
```

Brukes ved daglig sesongmønster i timesdata.

---

## 13. ACF og PACF

ACF og PACF brukes for å se autokorrelasjon og velge ARIMA-parametere.

```python id="mxe7gf"
plot_acf(df["load"].dropna(), lags=50)
plt.show()

plot_pacf(df["load"].dropna(), lags=50)
plt.show()
```

Brukes til å få en indikasjon på:

```text id="t2af7e"
p → AR-ledd
d → differencing
q → MA-ledd
```

I praksis testes ofte flere kombinasjoner.

---

## 14. Train/test split for tidsserier

Ikke bruk random split. Bruk dato.

```python id="yonzat"
train_start = "2014-11-01 00:00:00"
test_start = "2014-12-30 00:00:00"

train = df.loc[(df.index >= train_start) & (df.index < test_start), ["load"]]
test = df.loc[df.index >= test_start, ["load"]]

print(train.shape)
print(test.shape)
```

Visualiser split:

```python id="s1no0n"
train["load"].plot(figsize=(14, 5), label="train")
test["load"].plot(label="test")
plt.legend()
plt.show()
```

---

## 15. Scaling

Scaling brukes ofte ved ML-modeller som SVR, neural networks og LSTM. For ARIMA er det ikke alltid nødvendig, men kan brukes.

```python id="8ltwgj"
scaler = MinMaxScaler()

train_scaled = train.copy()
test_scaled = test.copy()

train_scaled["load"] = scaler.fit_transform(train[["load"]])
test_scaled["load"] = scaler.transform(test[["load"]])
```

Viktig:

```text id="zid8sa"
fit_transform på train
transform på test
```

Ikke fit scaler på hele datasettet før split.

---

## 16. Naive forecast baseline

En enkel baseline er å bruke forrige verdi som neste prediksjon.

```python id="elijrj"
test["naive_forecast"] = test["load"].shift(1)
```

For sesongbasert baseline kan man bruke samme time dagen før:

```python id="ch75fw"
test["seasonal_naive"] = df["load"].shift(24).loc[test.index]
```

Dette er viktig fordi avanserte modeller bør slå en enkel baseline.

---

## 17. ARIMA

ARIMA står for:

```text id="5airui"
AR → AutoRegressive
I  → Integrated
MA → Moving Average
```

Parametere:

```text id="uiftea"
p → antall lag i autoregressive delen
d → antall differencing-steg
q → antall moving average-ledd
```

Vanlig form:

```python id="vuz7w0"
order = (p, d, q)
```

Eksempel:

```python id="v7hp08"
order = (2, 1, 0)
```

---

## 18. SARIMA / SARIMAX

SARIMA brukes når serien har sesongmønster.

```text id="xc4vew"
order = (p, d, q)
seasonal_order = (P, D, Q, s)
```

Der:

```text id="spbrkn"
P → seasonal AR
D → seasonal differencing
Q → seasonal MA
s → sesonglengde
```

Eksempler på `s`:

```text id="ebpd48"
24  → timebasert data med daglig sesong
7   → daglig data med ukentlig sesong
12  → månedlig data med årlig sesong
```

---

## 19. SARIMAX standardoppsett

```python id="hwzx6f"
order = (2, 1, 0)
seasonal_order = (1, 1, 0, 24)

model = SARIMAX(
    train_scaled["load"],
    order=order,
    seasonal_order=seasonal_order
)

results = model.fit()

print(results.summary())
```

Forecast:

```python id="yrsgt1"
HORIZON = len(test_scaled)

forecast = results.forecast(steps=HORIZON)
```

Inverse scaling:

```python id="foad9e"
forecast_inv = scaler.inverse_transform(
    np.array(forecast).reshape(-1, 1)
)

test_inv = scaler.inverse_transform(test_scaled[["load"]])
```

---

## 20. Plot forecast mot actual

```python id="k3xwr9"
plt.figure(figsize=(14, 5))
plt.plot(test.index, test_inv, label="Actual")
plt.plot(test.index, forecast_inv, label="Forecast")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Load")
plt.title("Forecast vs Actual")
plt.show()
```

---

## 21. Walk-forward validation

Walk-forward validation er ofte bedre enn én statisk forecast.

Ideen:

```text id="tju1xk"
1. Tren på historikk
2. Prediker neste steg
3. Legg faktisk verdi til historikken
4. Flytt vinduet videre
5. Gjenta
```

Eksempel:

```python id="t20qep"
history = list(train_scaled["load"])
predictions = []

order = (2, 1, 0)
seasonal_order = (1, 1, 0, 24)

for t in range(len(test_scaled)):
    model = SARIMAX(
        history,
        order=order,
        seasonal_order=seasonal_order
    )

    model_fit = model.fit(disp=False)

    yhat = model_fit.forecast(steps=1)[0]
    predictions.append(yhat)

    actual = test_scaled["load"].iloc[t]
    history.append(actual)
```

Inverse transform:

```python id="juvstx"
predictions_inv = scaler.inverse_transform(
    np.array(predictions).reshape(-1, 1)
)

actual_inv = scaler.inverse_transform(test_scaled[["load"]])
```

---

## 22. Multi-step forecasting

Multi-step forecast betyr at man predikerer flere steg frem.

```python id="c8cmth"
HORIZON = 3

forecast = results.forecast(steps=HORIZON)
```

Eksempel:

```text id="b8itjh"
t+1
t+2
t+3
```

Typisk blir prediksjonen mer usikker jo lenger frem man predikerer.

---

## 23. Lage supervised dataset fra tidsserie

ML-modeller som SVR, Random Forest og XGBoost trenger X og y.

Man lager lag-features:

```python id="1kj02z"
def create_lag_features(df, target_col, lags):
    data = df.copy()

    for lag in lags:
        data[f"lag_{lag}"] = data[target_col].shift(lag)

    data = data.dropna()
    return data
```

Bruk:

```python id="i7zodk"
lagged_df = create_lag_features(df, "load", lags=[1, 2, 3, 24, 48, 168])

lagged_df.head()
```

Split:

```python id="lgbhri"
X = lagged_df.drop(columns=["load"])
y = lagged_df["load"]
```

---

## 24. Legge til time features

Ofte nyttig for sesongmønstre.

```python id="h7y6dm"
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month
df["dayofyear"] = df.index.dayofyear
df["weekend"] = (df.index.dayofweek >= 5).astype(int)
```

Eksempel:

```python id="t2qdbn"
features = ["hour", "dayofweek", "month", "weekend", "lag_1", "lag_24", "lag_168"]
```

---

## 25. SVR for tidsserier

SVR kan brukes når man gjør tidsserien om til supervised learning med lag-features.

```python id="fht2fp"
from sklearn.svm import SVR
```

Standardoppsett:

```python id="nm5v8x"
model = SVR(
    kernel="rbf",
    C=10,
    gamma=0.5,
    epsilon=0.05
)
```

Viktige parametere:

```text id="spf23a"
kernel   → type funksjon, ofte "rbf"
C        → hvor mye modellen straffer feil
gamma    → hvor lokal/fleksibel modellen er
epsilon  → margin rundt prediksjonen
```

---

## 26. SVR med timesteps

Lag sekvenser:

```python id="067eje"
train_data = train_scaled.values
test_data = test_scaled.values

timesteps = 5

train_ts = np.array([
    [j for j in train_data[i:i + timesteps]]
    for i in range(0, len(train_data) - timesteps + 1)
])[:, :, 0]

test_ts = np.array([
    [j for j in test_data[i:i + timesteps]]
    for i in range(0, len(test_data) - timesteps + 1)
])[:, :, 0]
```

X og y:

```python id="qys1ay"
X_train = train_ts[:, :timesteps - 1]
y_train = train_ts[:, timesteps - 1]

X_test = test_ts[:, :timesteps - 1]
y_test = test_ts[:, timesteps - 1]
```

Tren SVR:

```python id="nxp1w6"
model = SVR(kernel="rbf", C=10, gamma=0.5, epsilon=0.05)

model.fit(X_train, y_train)
```

Prediker:

```python id="g4d5as"
y_pred = model.predict(X_test).reshape(-1, 1)
```

Inverse transform:

```python id="cov3ev"
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))
```

---

## 27. Evaluering

Vanlige metrikker:

```text id="iur0np"
MAE   → mean absolute error
RMSE  → root mean squared error
MAPE  → mean absolute percentage error
```

Kode:

```python id="q2d69p"
mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mape = mean_absolute_percentage_error(y_test_inv, y_pred_inv) * 100

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
```

Tolkning:

```text id="n87jqx"
MAE  → gjennomsnittlig absolutt feil i samme enhet som target
RMSE → straffer store feil mer enn MAE
MAPE → prosentvis feil
```

Eksempel:

```text id="olctmb"
MAPE = 10 betyr at modellen i snitt bommer med ca. 10 %
```

---

## 28. Plot actual vs predicted

```python id="af8g2z"
plt.figure(figsize=(14, 5))
plt.plot(y_test_inv, label="Actual")
plt.plot(y_pred_inv, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted")
plt.show()
```

---

## 29. TimeSeriesSplit

For klassiske ML-modeller kan man bruke `TimeSeriesSplit`.

```python id="52omp6"
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

Dette bevarer tidsrekkefølgen.

---

## 30. Sammenligne flere modeller

Eksempel med lag-features:

```python id="5cxhii"
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "SVR": SVR(kernel="rbf", C=10, gamma=0.5, epsilon=0.05)
}

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    print(f"\n{name}")
    print("MAE:", mae)
    print("RMSE:", rmse)
    print("MAPE:", mape)
```

---

## 31. Typiske modeller

```text id="ehnyo4"
Naive forecast
→ enkel baseline

Seasonal naive
→ god baseline ved sterk sesong

ARIMA
→ klassisk univariat tidsseriemodell

SARIMA/SARIMAX
→ ARIMA med sesong og eventuelt eksterne variabler

SVR
→ ML-regresjon med lag-features

Random Forest / Gradient Boosting
→ sterke ML-baselines med lag og kalenderfeatures

XGBoost / LightGBM
→ ofte sterke på tabulære tidsseriefeatures

LSTM / GRU
→ deep learning for sekvensdata

Prophet
→ praktisk modell for trend og seasonality
```

---

## 32. Når brukes hva?

```text id="p6vk7r"
Naive / seasonal naive
→ alltid som baseline

ARIMA
→ én variabel, relativt klassisk statistisk forecasting

SARIMA
→ én variabel med tydelig sesong

SARIMAX
→ som SARIMA, men med eksterne forklaringsvariabler

SVR
→ ikke-lineær modell med korte sekvenser eller lag-features

Random Forest / XGBoost
→ når du lager lag-features og kalenderfeatures

LSTM
→ når sekvensmønstre er komplekse og du har nok data
```

---

## 33. Vanlige feil

```text id="c43xj1"
1. Random train/test split
Gir data leakage fordi fremtiden blandes inn i treningen.

2. Fitte scaler på hele datasettet
Scaler må fit_transformes på train og kun transformes på test.

3. Ikke bruke baseline
Modellen bør sammenlignes mot naive eller seasonal naive forecast.

4. Ignorere seasonality
Mange tidsserier har daglig, ukentlig eller årlig mønster.

5. Feil sesonglengde
Timesdata med daglig mønster bruker ofte 24, ikke 7 eller 12.

6. Bare se på én metrikk
Bruk gjerne MAE, RMSE og MAPE sammen.

7. For lang forecast horizon
Flere steg frem gir ofte mer usikre prediksjoner.

8. Overfitting på kort testperiode
Testperioden bør representere reell fremtidig variasjon.
```

---

## 34. Komplett baseline med SARIMA

```python id="udt9cw"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv("data/energy.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df = df.set_index("timestamp")

target = "load"

train = df.loc["2014-11-01":"2014-12-29", [target]]
test = df.loc["2014-12-30":, [target]]

scaler = MinMaxScaler()

train_scaled = train.copy()
test_scaled = test.copy()

train_scaled[target] = scaler.fit_transform(train[[target]])
test_scaled[target] = scaler.transform(test[[target]])

order = (2, 1, 0)
seasonal_order = (1, 1, 0, 24)

model = SARIMAX(
    train_scaled[target],
    order=order,
    seasonal_order=seasonal_order
)

results = model.fit(disp=False)

forecast = results.forecast(steps=len(test_scaled))

forecast_inv = scaler.inverse_transform(
    np.array(forecast).reshape(-1, 1)
)

actual_inv = scaler.inverse_transform(test_scaled[[target]])

mae = mean_absolute_error(actual_inv, forecast_inv)
rmse = np.sqrt(mean_squared_error(actual_inv, forecast_inv))
mape = mean_absolute_percentage_error(actual_inv, forecast_inv) * 100

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

plt.figure(figsize=(14, 5))
plt.plot(test.index, actual_inv, label="Actual")
plt.plot(test.index, forecast_inv, label="Forecast")
plt.legend()
plt.title("SARIMA forecast")
plt.show()
```

---

## 35. Komplett baseline med lag-features og ML

```python id="zpe60w"
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

df = pd.read_csv("data/energy.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp")
df = df.set_index("timestamp")

target = "load"

df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month
df["weekend"] = (df.index.dayofweek >= 5).astype(int)

for lag in [1, 2, 3, 24, 48, 168]:
    df[f"lag_{lag}"] = df[target].shift(lag)

df = df.dropna()

features = [
    "hour",
    "dayofweek",
    "month",
    "weekend",
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_24",
    "lag_48",
    "lag_168"
]

train = df.loc[: "2014-12-29"]
test = df.loc["2014-12-30":]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)

plt.figure(figsize=(14, 5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_pred, label="Predicted")
plt.legend()
plt.title("ML forecast with lag features")
plt.show()
```

---

## 36. Kort oppsummert

Den viktigste praktiske tidsserie-pipelinen:

```text id="g9gcrd"
Datetime index
→ plot
→ inspect trend/seasonality
→ split by time
→ baseline forecast
→ SARIMA or lag-based ML model
→ evaluate with MAE/RMSE/MAPE
```

For ARIMA/SARIMA:

```python id="lkn5n4"
SARIMAX(
    train,
    order=(p, d, q),
    seasonal_order=(P, D, Q, s)
)
```

For ML-basert tidsserie:

```text id="04vmff"
target
→ lag_1, lag_24, lag_168
→ hour, dayofweek, month
→ RandomForest / XGBoost / SVR
```

Viktigste regel:

```text id="fy76h2"
Aldri bland fremtidige observasjoner inn i treningsdata.
```

Start alltid med en enkel baseline, sjekk sesongmønster, og gå videre til SARIMA eller lag-baserte ML-modeller hvis baseline ikke er god nok.