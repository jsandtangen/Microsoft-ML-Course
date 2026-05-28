# Regresjon - teknisk README

Dette dokumentet oppsummerer en vanlig teknisk arbeidsflyt for regresjon i Python. Regresjon brukes når målet er å predikere en numerisk verdi, for eksempel pris, temperatur, salg eller forbruk.

## Vanlige biblioteker

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

Vanlige biblioteker:

* `pandas` brukes til å lese, rydde og analysere datasett
* `numpy` brukes til numeriske beregninger
* `matplotlib` brukes til visualisering
* `scikit-learn` brukes til modelltrening, splitting av data og evaluering

---

## Lese inn data

```python
df = pd.read_csv("data.csv")

df.head()
df.info()
df.describe()
```

Dette brukes for å få oversikt over datasettet, kolonner, datatyper og statistikk.

---

## Sjekke manglende verdier

```python
df.isnull().sum()
```

Hvis datasettet har manglende verdier, kan man for eksempel fjerne dem:

```python
df = df.dropna()
```

Eller fylle dem med en verdi:

```python
df["kolonne"] = df["kolonne"].fillna(df["kolonne"].mean())
```

---

## Velge features og target

```python
X = df[["feature_1", "feature_2", "feature_3"]]
y = df["target"]
```

* `X` er inputvariablene modellen skal bruke
* `y` er verdien modellen skal predikere

Eksempel:

```python
X = df[["areal", "soverom", "byggeaar"]]
y = df["pris"]
```

---

## Dele data i trening og test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

* `X_train` og `y_train` brukes til å trene modellen
* `X_test` og `y_test` brukes til å evaluere modellen

---

## Trene en lineær regresjonsmodell

```python
model = LinearRegression()

model.fit(X_train, y_train)
```

`fit()` trener modellen på treningsdataene.

---

## Lage prediksjoner

```python
predictions = model.predict(X_test)
```

`predict()` brukes for å lage prediksjoner på testdata.

---

## Evaluere modellen

```python
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
```

Vanlige evalueringsmål:

* `MAE`: gjennomsnittlig absolutt feil
* `MSE`: gjennomsnittlig kvadrert feil
* `RMSE`: kvadratroten av MSE
* `R²`: hvor mye av variasjonen modellen forklarer

---

## Visualisere faktisk verdi mot predikert verdi

```python
plt.scatter(y_test, predictions)
plt.xlabel("Faktiske verdier")
plt.ylabel("Predikerte verdier")
plt.title("Faktiske vs. predikerte verdier")
plt.show()
```

Hvis modellen treffer godt, bør punktene ligge omtrent langs en diagonal linje.

---

## Visualisere residualer

```python
residuals = y_test - predictions

plt.scatter(predictions, residuals)
plt.axhline(y=0)
plt.xlabel("Predikerte verdier")
plt.ylabel("Residualer")
plt.title("Residualplot")
plt.show()
```

Residualer er forskjellen mellom faktisk og predikert verdi.

Et residualplot brukes for å se om modellen bommer tilfeldig, eller om det finnes tydelige mønstre i feilene.

---

## Polynomial Regression

Polynomial Regression kan brukes hvis sammenhengen ikke er lineær.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

poly_model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

poly_model.fit(X_train, y_train)

poly_predictions = poly_model.predict(X_test)
```

Dette lager nye features basert på potenser av de eksisterende variablene.

---

## Skalering

Noen modeller fungerer bedre når dataene skaleres.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

Viktig:

* Bruk `fit_transform()` på treningsdata
* Bruk `transform()` på testdata

Dette hindrer data leakage.

---

## Pipeline

Pipeline gjør at preprocessing og modelltrening samles i én struktur.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

pipeline = make_pipeline(
    StandardScaler(),
    Ridge()
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
```

Pipeline er nyttig når man kombinerer flere steg, for eksempel skalering og modelltrening.

---

## Eksempel med Random Forest Regressor

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

Random Forest kan fange opp mer komplekse og ikke-lineære sammenhenger enn lineær regresjon.

---

## Full enkel regresjonsflyt

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("data.csv")

X = df[["feature_1", "feature_2", "feature_3"]]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

plt.scatter(y_test, predictions)
plt.xlabel("Faktiske verdier")
plt.ylabel("Predikerte verdier")
plt.title("Faktiske vs. predikerte verdier")
plt.show()
```

---