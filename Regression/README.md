# Regresjon i maskinlæring

Regresjon er en sentral metode innen maskinlæring som brukes når målet er å predikere en numerisk verdi. I stedet for å plassere data i kategorier, forsøker en regresjonsmodell å finne sammenhenger mellom inputvariabler og en kontinuerlig outputverdi.

Regresjon brukes ofte til å forutsi verdier som pris, temperatur, salg, strømforbruk, etterspørsel eller andre tallbaserte resultater.

Eksempler på regresjonsproblemer:

- Forutsi boligpris basert på areal, beliggenhet og antall soverom
- Forutsi temperatur basert på historiske værdata
- Forutsi salg neste måned basert på tidligere salg
- Forutsi strømforbruk basert på tid, vær og sesong
- Forutsi prisutvikling basert på dato, produktkategori eller marked

Regresjon tilhører vanligvis **supervised learning**, fordi modellen trenes på data der riktig svar allerede er kjent. Modellen får inputdata `X` og en målvariabel `y`, og prøver å lære sammenhengen mellom disse.

---

## Supervised learning

I supervised learning trener vi en modell på eksempler der fasiten allerede finnes.

Vanlig struktur:

```python
X = features
y = target
```

- `X` er inputvariablene modellen bruker for å lære.
- `y` er verdien modellen skal lære å predikere.

Eksempel:

```python
X = df[["areal", "soverom", "byggeaar"]]
y = df["pris"]
```

Her prøver modellen å lære hvordan areal, antall soverom og byggeår henger sammen med boligpris.

---

## Hva brukes regresjon til?

Regresjon brukes når svaret vi ønsker å predikere er et tall.

Det kan for eksempel være:

- Pris
- Temperatur
- Salg
- Forbruk
- Etterspørsel
- Tid
- Poengsum
- Inntekt

Dette skiller regresjon fra klassifikasjon, hvor målet er å predikere en kategori.

Eksempler på klassifikasjon:

- Spam eller ikke spam
- Syk eller frisk
- Ja eller nei
- Kunde kjøper eller kjøper ikke

Eksempel på regresjon:

```text
Hva blir boligprisen?
```

Eksempel på klassifikasjon:

```text
Blir boligen solgt eller ikke?
```

---

## Regresjon vs. klassifikasjon

Forskjellen mellom regresjon og klassifikasjon handler først og fremst om hva modellen skal predikere.

| Type | Predikerer | Eksempel |
|---|---|---|
| Regresjon | Tallverdi | Pris, temperatur, salg |
| Klassifikasjon | Klasse/kategori | Spam/ikke spam, ja/nei, sykdom/ikke sykdom |

---

## Typiske regresjonsmodeller

Noen vanlige regresjonsmodeller er:

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

Hvilken modell som passer best, avhenger av datasettet, problemstillingen og hvor komplekse sammenhengene i dataene er.

---

# Lineær regresjon

Lineær regresjon er en av de enkleste og mest grunnleggende formene for regresjon.

Målet er å finne en rett linje som best mulig beskriver sammenhengen mellom inputvariabelen og målvariabelen.

En enkel lineær regresjon kan beskrives slik:

```text
y = a + bx
```

Der:

- `y` er verdien vi ønsker å predikere
- `x` er inputvariabelen
- `a` er konstantleddet, altså verdien av `y` når `x = 0`
- `b` er stigningstallet, altså hvor mye `y` endrer seg når `x` øker med 1

Eksempel:

```text
boligpris = a + b * areal
```

Hvis stigningstallet er positivt, betyr det at boligprisen øker når arealet øker.

Hvis stigningstallet er negativt, betyr det at målverdien synker når inputverdien øker.

---

## Enkel lineær regresjon

Enkel lineær regresjon bruker én inputvariabel for å predikere én målverdi.

Eksempel:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X = df[["areal"]]
y = df["pris"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

Her prøver modellen å predikere pris basert på areal alene.

---

## Multippel lineær regresjon

Multippel lineær regresjon bruker flere inputvariabler samtidig.

Eksempel:

```python
X = df[["areal", "soverom", "byggeaar", "avstand_sentrum"]]
y = df["pris"]
```

Dette er ofte mer realistisk enn enkel lineær regresjon, fordi virkelige problemer sjelden avhenger av bare én faktor.

En boligpris påvirkes for eksempel ikke bare av areal, men også av beliggenhet, standard, byggeår, antall rom og markedssituasjon.

---

## Polynomial Regression

Polynomial Regression brukes når sammenhengen i dataene ikke passer godt med en rett linje.

Noen ganger er forholdet mellom `X` og `y` kurvet. Da kan en lineær modell bli for enkel.

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

Polynomial Regression lager nye features basert på potenser av de eksisterende variablene.

Eksempel:

```text
x
x²
x³
```

Dette gjør at modellen kan fange opp mer komplekse mønstre.

Samtidig må man være forsiktig. Hvis graden blir for høy, kan modellen begynne å tilpasse seg treningsdataene for mye. Dette kalles **overfitting**.

---

## Logistic Regression

Selv om navnet inneholder ordet "regression", brukes Logistic Regression vanligvis til klassifikasjon, ikke vanlig numerisk regresjon.

Logistic Regression brukes ofte når målet er å predikere en binær kategori.

Eksempler:

- Ja eller nei
- Syk eller frisk
- Spam eller ikke spam
- Kunde kjøper eller kjøper ikke

I stedet for å predikere et vanlig tall, predikerer Logistic Regression en sannsynlighet mellom 0 og 1.

Eksempel:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X = df[["feature_1", "feature_2"]]
y = df["kategori"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

Hvis sannsynligheten er over en bestemt grense, ofte `0.5`, klassifiseres observasjonen som klasse 1. Hvis den er under grensen, klassifiseres den som klasse 0.

---

# Arbeidsflyt i et regresjonsprosjekt

En vanlig arbeidsflyt for regresjon kan se slik ut:

1. Definer problemstillingen
2. Samle inn data
3. Utforsk datasettet
4. Rens og klargjør data
5. Velg relevante features
6. Del data i treningsdata og testdata
7. Tren modellen
8. Evaluer modellen
9. Juster modellen
10. Bruk modellen på nye data

---

## 1. Definer problemstillingen

Før man bygger en modell, må man vite hva man faktisk prøver å finne ut.

En god problemstilling kan være:

```text
Kan vi predikere boligpris basert på areal, antall soverom og byggeår?
```

En dårligere problemstilling kan være:

```text
Kan vi bruke maskinlæring på boligdata?
```

Jo mer konkret spørsmålet er, desto lettere blir det å velge riktig modell, riktige features og riktig evalueringsmetode.

---

## 2. Utforsk datasettet

Før man trener en modell, bør man forstå datasettet.

Vanlige ting å sjekke:

```python
df.head()
df.info()
df.describe()
df.isnull().sum()
```

Dette gir oversikt over:

- hvilke kolonner datasettet har
- hvilke datatyper kolonnene har
- om det finnes manglende verdier
- hvordan verdiene er fordelt
- om noen kolonner inneholder rare eller uventede verdier

---

## 3. Rens data

Data er sjelden klart til bruk rett fra kilden. Ofte må man rydde i datasettet før modellen kan trenes.

Vanlige steg:

- Fjerne unødvendige kolonner
- Håndtere manglende verdier
- Konvertere tekst til tall
- Fjerne eller håndtere outliers
- Lage nye nyttige kolonner
- Standardisere eller normalisere verdier ved behov

Eksempel på å fjerne rader med manglende verdier:

```python
df = df.dropna()
```

Eksempel på å lage en ny kolonne:

```python
df["pris"] = (df["lav_pris"] + df["høy_pris"]) / 2
```

---

## 4. Visualiser data

Visualisering er viktig for å forstå mønstre i dataene.

Eksempel med scatterplot:

```python
import matplotlib.pyplot as plt

plt.scatter(df["areal"], df["pris"])
plt.xlabel("Areal")
plt.ylabel("Pris")
plt.title("Sammenheng mellom areal og pris")
plt.show()
```

Et scatterplot kan hjelpe deg med å se om det finnes en tydelig sammenheng mellom to variabler.

Hvis punktene omtrent følger en rett linje, kan lineær regresjon være et godt utgangspunkt.

Hvis punktene følger en kurve, kan Polynomial Regression eller mer fleksible modeller være bedre.

---

## 5. Velg features og target

I maskinlæring skiller vi ofte mellom `X` og `y`.

```python
X = df[["feature_1", "feature_2", "feature_3"]]
y = df["target"]
```

- `X` er forklaringsvariablene.
- `y` er målvariabelen.

Eksempel:

```python
X = df[["areal", "soverom", "byggeaar"]]
y = df["pris"]
```

Her skal modellen lære å predikere `pris` basert på de tre inputvariablene.

---

## 6. Del data i training set og test set

For å evaluere modellen riktig, deler man datasettet i treningsdata og testdata.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

- Treningsdata brukes til å trene modellen.
- Testdata brukes til å evaluere modellen på data den ikke har sett før.

Dette er viktig fordi vi ikke bare vil vite hvor godt modellen husker treningsdataene. Vi vil vite hvor godt den generaliserer til nye data.

---

## 7. Tren modellen

Når dataene er klare, kan modellen trenes.

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

`fit()` betyr at modellen lærer sammenhengen mellom `X_train` og `y_train`.

---

## 8. Lag prediksjoner

Når modellen er trent, kan den brukes til å predikere verdier på nye data.

```python
predictions = model.predict(X_test)
```

`predictions` inneholder modellens estimerte verdier for testdataene.

---

# Evaluering av regresjonsmodeller

Etter at modellen har laget prediksjoner, må vi sjekke hvor gode de er.

Vanlige evalueringsmål for regresjon:

- MAE
- MSE
- RMSE
- R²

---

## MAE - Mean Absolute Error

MAE måler gjennomsnittlig absolutt feil mellom faktisk verdi og predikert verdi.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, predictions)

print(mae)
```

Hvis MAE er 20 000 i en boligprismodell, betyr det at modellen i snitt bommer med 20 000 kroner.

MAE er lett å tolke fordi den bruker samme enhet som målvariabelen.

---

## MSE - Mean Squared Error

MSE måler gjennomsnittet av de kvadrerte feilene.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)

print(mse)
```

Siden feilene kvadreres, straffer MSE store feil hardere enn MAE.

Dette kan være nyttig hvis store feil er ekstra alvorlige.

---

## RMSE - Root Mean Squared Error

RMSE er kvadratroten av MSE.

```python
import numpy as np
from sklearn.metrics import mean_squared_error

rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(rmse)
```

RMSE er ofte mer tolkbar enn MSE, fordi den er i samme enhet som målvariabelen.

Hvis modellen predikerer pris, vil RMSE også være i kroner.

---

## R² - Coefficient of Determination

R² forklarer hvor stor andel av variasjonen i målvariabelen modellen klarer å forklare.

```python
r2 = model.score(X_test, y_test)

print(r2)
```

R² kan tolkes omtrent slik:

```text
R² = 1.0   -> perfekt modell
R² = 0.0   -> modellen er ikke bedre enn å bare gjette gjennomsnittet
R² < 0.0   -> modellen er dårligere enn å gjette gjennomsnittet
```

Eksempel:

```text
R² = 0.80
```

Dette betyr at modellen forklarer omtrent 80 % av variasjonen i målvariabelen.

En lav R² betyr at modellen ikke fanger opp sammenhengen i dataene særlig godt.

Det kan skyldes at:

- Feil features er brukt
- Sammenhengen ikke er lineær
- Datasettet inneholder mye støy
- Det mangler viktige variabler
- Modellen er for enkel
- Dataene er dårlig renset

---

## Korrelasjon

Korrelasjon sier noe om hvor sterkt to variabler henger sammen.

Eksempel:

```python
df["areal"].corr(df["pris"])
```

Korrelasjon ligger vanligvis mellom -1 og 1.

```text
1     -> sterk positiv sammenheng
0     -> ingen tydelig lineær sammenheng
-1    -> sterk negativ sammenheng
```

En positiv korrelasjon betyr at når én variabel øker, øker ofte den andre også.

En negativ korrelasjon betyr at når én variabel øker, synker ofte den andre.

Viktig: Korrelasjon betyr ikke nødvendigvis årsakssammenheng. To variabler kan henge sammen uten at den ene direkte forårsaker den andre.

---

## Residualer

Residualer er forskjellen mellom faktisk verdi og predikert verdi.

```text
residual = faktisk verdi - predikert verdi
```

Hvis modellen predikerer 300, men faktisk verdi er 350, er residualen 50.

Residualer brukes for å forstå hvor modellen bommer.

En god regresjonsmodell bør ha residualer som er relativt små og tilfeldig fordelt.

Hvis residualene viser et tydelig mønster, kan det bety at modellen ikke fanger opp en viktig sammenheng i dataene.

---

# Overfitting og underfitting

Når man trener regresjonsmodeller, er det viktig å forstå forskjellen mellom overfitting og underfitting.

---

## Underfitting

Underfitting skjer når modellen er for enkel til å fange opp mønstrene i dataene.

Tegn på underfitting:

- Dårlig score på treningsdata
- Dårlig score på testdata
- Modellen bommer systematisk
- Modellen klarer ikke å følge mønstrene i dataene

Eksempel:

En rett linje brukes på data som egentlig følger en tydelig kurve.

---

## Overfitting

Overfitting skjer når modellen lærer treningsdataene for godt, inkludert støy og tilfeldigheter.

Tegn på overfitting:

- Veldig god score på treningsdata
- Mye dårligere score på testdata
- Modellen generaliserer dårlig til nye data

Eksempel:

En veldig kompleks Polynomial Regression følger nesten alle datapunktene perfekt, men fungerer dårlig på nye observasjoner.

---

# Feature encoding

Maskinlæringsmodeller jobber best med tall. Derfor må tekstverdier ofte gjøres om til numeriske verdier.

Eksempler på kategoriske verdier:

```text
By: Oslo, Bergen, Trondheim
Produkttype: liten, medium, stor
Farge: rød, blå, grønn
```

For slike variabler kan man bruke encoding.

---

## One-hot encoding

One-hot encoding gjør kategorier om til egne kolonner med 0 og 1.

Eksempel:

```python
pd.get_dummies(df["by"])
```

Hvis en kolonne inneholder byene Oslo, Bergen og Trondheim, kan den gjøres om til:

```text
by_Oslo
by_Bergen
by_Trondheim
```

Hver rad får 1 i kolonnen som passer, og 0 i de andre.

Dette er nyttig for kategorier uten naturlig rekkefølge.

---

## Ordinal encoding

Ordinal encoding brukes når kategoriene har en naturlig rekkefølge.

Eksempel:

```text
small < medium < large
```

Da kan man kode dem som:

```text
small = 0
medium = 1
large = 2
```

Dette passer når rekkefølgen faktisk betyr noe.

Man bør være forsiktig med ordinal encoding på kategorier uten rekkefølge, fordi modellen kan tolke tallene som om noen kategorier er "større" eller "bedre" enn andre.

---

# Skalering av data

Noen modeller fungerer bedre når numeriske variabler er skalert.

Skalering betyr at man justerer verdiene slik at de ligger på en mer sammenlignbar skala.

Eksempel:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
```

Lineær regresjon krever ikke alltid skalering, men det kan være nyttig når man bruker modeller som er følsomme for størrelsesforskjeller mellom variabler.

Eksempler på modeller der skalering ofte er viktig:

- KNN
- SVM
- Neural Networks
- Ridge Regression
- Lasso Regression
- Logistic Regression

---

# Pipeline

En pipeline brukes for å samle flere steg i én samlet arbeidsflyt.

Eksempel:

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

Fordelen med pipeline er at preprocessing og modelltrening blir mer ryddig og mindre feilutsatt.

Pipeline er spesielt nyttig når man kombinerer flere steg, for eksempel:

- Encoding
- Skalering
- Polynomial features
- Modelltrening

---

# Eksempel på komplett regresjonsflyt

```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Velg features og target
X = df[["feature_1", "feature_2", "feature_3"]]
y = df["target"]

# Del data i trening og test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Lag og tren modell
model = LinearRegression()
model.fit(X_train, y_train)

# Lag prediksjoner
predictions = model.predict(X_test)

# Evaluer modellen
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = model.score(X_test, y_test)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)
```

---

# Hvordan tolke resultatene?

Når modellen er evaluert, bør man ikke bare se på én verdi. Det er ofte lurt å se på flere ting samtidig.

Eksempel:

```text
MAE = 2.1
RMSE = 2.8
R² = 0.06
```

Dette kan bety at modellen i snitt bommer med litt over 2 enheter, men at den likevel forklarer lite av variasjonen i dataene.

En lav R² betyr at modellen ikke bruker inputdataene særlig godt til å forklare målvariabelen. Da kan modellen i praksis ligge nær det å bare gjette gjennomsnittet.

Mulige tiltak kan være:

- Bruke flere relevante features
- Undersøke om sammenhengen er ikke-lineær
- Prøve Polynomial Regression
- Prøve andre regresjonsmodeller
- Rense data bedre
- Fjerne eller undersøke outliers
- Sjekke om datasettet er for lite
- Undersøke om målvariabelen er vanskelig å predikere med tilgjengelige data

---

# Når passer lineær regresjon?

Lineær regresjon passer godt når:

- Målet er å predikere en numerisk verdi
- Sammenhengen mellom variablene er omtrent lineær
- Man ønsker en enkel og tolkbar modell
- Man vil forstå hvordan ulike variabler påvirker målvariabelen

Lineær regresjon er ofte et godt førstevalg fordi den er enkel å forstå, rask å trene og lett å forklare.

---

# Når passer lineær regresjon dårlig?

Lineær regresjon passer dårligere når:

- Sammenhengen i dataene er sterkt ikke-lineær
- Datasettet har mange outliers
- Viktige variabler mangler
- Dataene har komplekse interaksjoner
- Residualene viser tydelige mønstre
- Modellen får lav R² og høy feil

I slike tilfeller kan man prøve mer fleksible modeller, som Random Forest, Gradient Boosting eller Polynomial Regression.

---

# Viktige ting å huske

- Regresjon brukes når målet er å predikere tallverdier.
- `X` er inputvariablene, og `y` er målvariabelen.
- Data bør utforskes og renses før modelltrening.
- Visualisering hjelper med å forstå sammenhenger.
- Train/test split brukes for å teste modellen på ukjente data.
- `fit()` trener modellen.
- `predict()` lager prediksjoner.
- MAE, RMSE og R² brukes ofte for å evaluere regresjonsmodeller.
- Lav feil betyr ikke alltid at modellen er god. Man bør også vurdere R² og kontekst.
- Høy R² betyr ikke automatisk at modellen er perfekt. Man må også sjekke overfitting og datakvalitet.
- Logistic Regression brukes vanligvis til klassifikasjon, selv om navnet inneholder regression.
- En god modell starter med et godt spørsmål og relevante data.

---

# Kort oppsummering

Regresjon handler om å predikere numeriske verdier basert på historiske data. Modellen lærer sammenhenger mellom inputvariabler og en målvariabel, og bruker denne læringen til å lage prediksjoner på nye data.

En typisk regresjonsprosess består av å forstå problemet, rense data, velge features, dele data i trening og test, trene modellen, lage prediksjoner og evaluere resultatene.

Lineær regresjon er et godt sted å starte fordi modellen er enkel, rask og lett å tolke. Hvis sammenhengene i dataene er mer komplekse, kan man prøve Polynomial Regression eller andre mer avanserte regresjonsmodeller.

Det viktigste er ikke bare å få en modell til å kjøre, men å forstå hva resultatene betyr. En regresjonsmodell bør alltid vurderes ut fra både feilmarginer, forklaringsgrad, datakvalitet og den praktiske problemstillingen modellen skal brukes til.