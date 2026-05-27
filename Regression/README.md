# Regresjon i maskinlæring

Regresjon er en metode innen supervised learning der målet er å predikere en kontinuerlig numerisk verdi. Modellen lærer en sammenheng mellom et sett med forklaringsvariabler, ofte kalt features, og en målvariabel, ofte kalt target.

Regresjon brukes når svaret ikke er en klasse eller kategori, men et tall. Typiske eksempler er prediksjon av boligpris, temperatur, salg, strømforbruk, etterspørsel eller andre målbare verdier.

I et regresjonsproblem har vi vanligvis følgende struktur:

```python
X = features
y = target
```

Der `X` inneholder variablene modellen bruker for å lære, mens `y` er verdien modellen skal predikere.

Eksempel:

```python
X = df[["areal", "soverom", "byggeaar"]]
y = df["pris"]
```

Her forsøker modellen å lære hvordan areal, antall soverom og byggeår henger sammen med boligpris.

---

## Regresjon vs. klassifikasjon

Forskjellen mellom regresjon og klassifikasjon ligger i hva modellen skal predikere.

| Problemtype | Predikerer | Eksempel |
|---|---|---|
| Regresjon | Kontinuerlig tallverdi | Pris, temperatur, salg |
| Klassifikasjon | Klasse/kategori | Spam/ikke spam, ja/nei, syk/frisk |

Regresjon svarer typisk på spørsmål som:

```text
Hva blir verdien?
```

Klassifikasjon svarer heller på:

```text
Hvilken kategori tilhører observasjonen?
```

---

## Vanlige regresjonsmodeller

Det finnes flere typer regresjonsmodeller, fra enkle og tolkbare modeller til mer fleksible modeller som kan fange opp komplekse mønstre.

Vanlige modeller er:

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- Neural Networks

Valg av modell avhenger av datasettet, problemstillingen, graden av ikke-linearitet og hvor viktig tolkbarhet er.

---

## Lineær regresjon

Lineær regresjon er en grunnleggende regresjonsmodell som antar en lineær sammenheng mellom inputvariablene og målvariabelen.

For én forklaringsvariabel kan modellen skrives som:

```text
y = a + bx
```

Der:

- `y` er målvariabelen
- `x` er inputvariabelen
- `a` er konstantleddet
- `b` er koeffisienten/stigningstallet

I praksis betyr koeffisienten hvor mye `y` forventes å endre seg når `x` øker med én enhet, gitt at modellen er lineær.

Eksempel:

```text
pris = a + b * areal
```

Hvis `b` er positiv, øker predikert pris når arealet øker. Hvis `b` er negativ, synker predikert verdi når inputverdien øker.

---

## Multippel lineær regresjon

I praktiske problemer bruker man ofte flere forklaringsvariabler samtidig. Dette kalles multippel lineær regresjon.

```python
X = df[["areal", "soverom", "byggeaar", "avstand_sentrum"]]
y = df["pris"]
```

Modellen forsøker da å estimere hvordan hver variabel bidrar til prediksjonen, kontrollert for de andre variablene i modellen.

Dette er mer realistisk enn enkel lineær regresjon, fordi de fleste fenomener påvirkes av flere faktorer samtidig.

---

## Polynomial Regression

Polynomial Regression brukes når sammenhengen mellom inputvariablene og målvariabelen ikke er godt beskrevet av en rett linje.

Ved å lage nye features som potenser av eksisterende variabler, kan modellen fange opp kurvede sammenhenger.

Eksempel:

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

model = make_pipeline(
    PolynomialFeatures(degree=2),
    LinearRegression()
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

En modell med `degree=2` inkluderer blant annet andregradsledd. Dette kan gi bedre tilpasning dersom dataene har en ikke-lineær struktur.

Samtidig øker risikoen for overfitting når modellen blir mer kompleks. Derfor bør polynomial regression alltid vurderes mot testdata, ikke bare treningsdata.

---

## Logistic Regression

Logistic Regression har ordet "regression" i navnet, men brukes vanligvis til klassifikasjon.

Modellen estimerer sannsynligheten for at en observasjon tilhører en bestemt klasse, ofte i et binært problem.

Eksempler:

- Kunde kjøper eller kjøper ikke
- Spam eller ikke spam
- Syk eller frisk
- Ja eller nei

Eksempel:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

Logistic Regression er derfor viktig å kjenne til, men bør skilles fra regresjonsmodeller som predikerer kontinuerlige verdier.

---

## Typisk arbeidsflyt i et regresjonsprosjekt

En vanlig prosess for regresjon er:

1. Definer problemstillingen
2. Utforsk datasettet
3. Rens og klargjør data
4. Velg features og target
5. Del data i trenings- og testsett
6. Tren modellen
7. Lag prediksjoner
8. Evaluer modellen
9. Juster modell eller features ved behov

Eksempel på en enkel regresjonsflyt:

```python
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
r2 = model.score(X_test, y_test)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
```

---

## Train/test split

For å evaluere en modell må dataene deles i treningsdata og testdata.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

Treningsdata brukes til å trene modellen, mens testdata brukes til å undersøke hvor godt modellen generaliserer til nye observasjoner.

Dette er viktig fordi en modell kan prestere godt på treningsdata uten nødvendigvis å fungere godt på nye data.

---

## Evaluering av regresjonsmodeller

Regresjonsmodeller evalueres ved å sammenligne faktiske verdier med predikerte verdier.

Vanlige evalueringsmål er:

- MAE
- MSE
- RMSE
- R²

---

## MAE

Mean Absolute Error måler gjennomsnittlig absolutt avvik mellom faktisk og predikert verdi.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)
```

MAE er lett å tolke fordi den har samme enhet som målvariabelen.

Hvis modellen predikerer boligpris og MAE er 50 000, betyr det at modellen i snitt bommer med 50 000 kroner.

---

## MSE og RMSE

Mean Squared Error måler gjennomsnittet av kvadrerte feil.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
```

RMSE er kvadratroten av MSE.

```python
rmse = np.sqrt(mean_squared_error(y_test, predictions))
```

RMSE er ofte mer tolkbar enn MSE fordi den har samme enhet som målvariabelen.

Siden feilene kvadreres før gjennomsnittet beregnes, straffer MSE og RMSE store feil hardere enn MAE.

---

## R²

R², også kalt coefficient of determination, beskriver hvor stor andel av variasjonen i målvariabelen modellen klarer å forklare.

```python
r2 = model.score(X_test, y_test)
```

R² tolkes ofte slik:

```text
R² = 1.0   -> perfekt forklaring
R² = 0.0   -> modellen forklarer ikke mer enn gjennomsnittet
R² < 0.0   -> modellen er dårligere enn å bruke gjennomsnittet
```

En lav R² betyr at modellen ikke fanger opp mye av variasjonen i dataene. Det kan skyldes svake features, ikke-lineære sammenhenger, støy, manglende variabler eller at modellen er for enkel.

Det er viktig å vurdere R² sammen med feilmarginer som MAE og RMSE. En modell kan ha lav feil i absolutte tall, men fortsatt forklare lite av variasjonen.

---

## Korrelasjon

Korrelasjon beskriver graden av lineær sammenheng mellom to variabler.

```python
df["feature"].corr(df["target"])
```

Korrelasjon ligger mellom -1 og 1.

```text
1     -> sterk positiv lineær sammenheng
0     -> ingen tydelig lineær sammenheng
-1    -> sterk negativ lineær sammenheng
```

Korrelasjon kan være nyttig i utforskende analyse, men bør ikke tolkes som årsakssammenheng. To variabler kan være korrelert uten at den ene forårsaker den andre.

---

## Residualer

Residualer er forskjellen mellom faktisk verdi og predikert verdi.

```text
residual = faktisk verdi - predikert verdi
```

Residualanalyse brukes for å forstå hvor modellen bommer.

En god regresjonsmodell bør ha residualer som er relativt små og uten tydelig systematisk mønster. Hvis residualene viser struktur, kan det tyde på at modellen mangler viktige variabler eller at sammenhengen ikke er godt modellert.

---

## Overfitting og underfitting

Overfitting og underfitting handler om hvor godt modellen balanserer tilpasning til treningsdata mot generalisering til nye data.

### Underfitting

Underfitting skjer når modellen er for enkel til å fange opp mønstrene i dataene.

Typiske tegn:

- Svak score på treningsdata
- Svak score på testdata
- Modellen bommer systematisk
- Modellen klarer ikke å fange opp reelle mønstre

### Overfitting

Overfitting skjer når modellen tilpasser seg treningsdataene for godt, inkludert støy og tilfeldigheter.

Typiske tegn:

- Svært god score på treningsdata
- Betydelig dårligere score på testdata
- Modellen generaliserer dårlig til nye data

Målet er ikke å lage en modell som passer treningsdataene perfekt, men en modell som fungerer godt på nye observasjoner.

---

## Feature encoding

Maskinlæringsmodeller krever som regel numeriske inputverdier. Derfor må kategoriske variabler ofte kodes om.

Eksempler på kategoriske variabler:

- By
- Produkttype
- Farge
- Størrelse
- Kategori

For kategorier uten naturlig rekkefølge brukes ofte one-hot encoding.

```python
pd.get_dummies(df["kategori"])
```

For kategorier med naturlig rekkefølge kan ordinal encoding brukes.

Eksempel:

```text
small < medium < large
```

Dette kan kodes som:

```text
small = 0
medium = 1
large = 2
```

Det er viktig å velge riktig encoding, fordi feil encoding kan gi modellen kunstige sammenhenger.

---

## Skalering

Skalering betyr at numeriske variabler justeres slik at de ligger på en sammenlignbar skala.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

Lineær regresjon krever ikke alltid skalering, men skalering er ofte viktig for modeller som påvirkes av avstand, gradienter eller regularisering.

Eksempler:

- KNN
- SVM
- Logistic Regression
- Ridge Regression
- Lasso Regression
- Neural Networks

---

## Pipeline

En pipeline samler preprocessing og modelltrening i én strukturert arbeidsflyt.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipeline = make_pipeline(
    StandardScaler(),
    LinearRegression()
)

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

Pipeline gjør koden ryddigere og reduserer risikoen for feil, spesielt når man kombinerer flere steg som encoding, skalering, feature engineering og modelltrening.

---

## Når passer lineær regresjon?

Lineær regresjon passer godt når:

- Målet er å predikere en numerisk verdi
- Sammenhengen mellom features og target er omtrent lineær
- Man ønsker en enkel og tolkbar modell
- Man vil bruke modellen som baseline
- Forklarbarhet er viktig

Lineær regresjon er ofte et godt startpunkt, selv om mer avanserte modeller senere kan gi bedre prediksjoner.

---

## Når bør man vurdere andre modeller?

Andre modeller kan være bedre når:

- Sammenhengene er tydelig ikke-lineære
- Det finnes komplekse interaksjoner mellom variabler
- Residualene viser systematiske mønstre
- Lineær regresjon gir lav forklaringskraft
- Prediksjonskvalitet er viktigere enn enkel tolkbarhet

Da kan modeller som Random Forest, Gradient Boosting, XGBoost eller Neural Networks være aktuelle.

---

## Viktige ting å huske

- Regresjon brukes for å predikere kontinuerlige tallverdier.
- `X` inneholder features, og `y` inneholder target.
- Datakvalitet og featurevalg er ofte viktigere enn selve modellen.
- Train/test split brukes for å måle generalisering.
- MAE og RMSE måler prediksjonsfeil.
- R² sier noe om hvor mye variasjon modellen forklarer.
- Residualer bør sjekkes for systematiske mønstre.
- Overfitting betyr at modellen lærer treningsdataene for godt.
- Underfitting betyr at modellen er for enkel.
- Lineær regresjon er en god baseline, men ikke alltid den beste modellen.
- Logistic Regression brukes hovedsakelig til klassifikasjon.

---
