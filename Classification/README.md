# Klassifikasjon - teknisk README

Dette dokumentet oppsummerer en vanlig teknisk arbeidsflyt for klassifikasjon i Python. Klassifikasjon brukes når målet er å predikere en klasse eller kategori, for eksempel spam/ikke spam, kjøper/ikke kjøper eller syk/frisk.

## Vanlige biblioteker

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

Vanlige biblioteker:

* `pandas` brukes til å lese og bearbeide datasett
* `numpy` brukes til numeriske beregninger
* `matplotlib` brukes til visualisering
* `scikit-learn` brukes til modelltrening, splitting og evaluering

---

## Lese inn data

```python
df = pd.read_csv("data.csv")

df.head()
df.info()
df.describe()
```

Dette gir oversikt over datasettet, kolonner, datatyper og statistikk.

---

## Sjekke klassefordeling

```python
df["target"].value_counts()
```

Det er viktig å sjekke om klassene er balanserte.

For prosentvis fordeling:

```python
df["target"].value_counts(normalize=True)
```

---

## Visualisere klassefordeling

```python
df["target"].value_counts().plot(kind="bar")

plt.xlabel("Klasse")
plt.ylabel("Antall")
plt.title("Klassefordeling")
plt.show()
```

Dette gir rask oversikt over om datasettet har skjev klassefordeling.

---

## Velge features og target

```python
X = df[["feature_1", "feature_2", "feature_3"]]
y = df["target"]
```

* `X` er inputvariablene modellen skal bruke
* `y` er klassen modellen skal predikere

Eksempel:

```python
X = df[["alder", "inntekt", "tidligere_kjop"]]
y = df["kjoper"]
```

---

## Encode kategoriske variabler

Mange modeller krever numeriske inputverdier. Tekstvariabler må derfor ofte kodes om.

```python
X = pd.get_dummies(X, drop_first=True)
```

Dette gjør kategoriske kolonner om til numeriske dummy-variabler.

Eksempel:

```python
X = pd.get_dummies(df[["by", "produkttype", "alder"]], drop_first=True)
y = df["target"]
```

---

## Dele data i trening og test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Ved klassifikasjon er `stratify=y` ofte nyttig, spesielt ved ubalanserte klasser. Det gjør at klassefordelingen blir omtrent lik i trenings- og testsettet.

---

## Trene en Logistic Regression-modell

```python
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)
```

`fit()` trener modellen på treningsdataene.

`max_iter=1000` brukes ofte for å gi modellen nok iterasjoner til å konvergere.

---

## Lage prediksjoner

```python
predictions = model.predict(X_test)
```

Dette gir predikerte klasser.

Eksempel:

```text
0, 1, 0, 0, 1
```

---

## Predikerte sannsynligheter

Noen modeller kan også gi sannsynligheter for hver klasse.

```python
probabilities = model.predict_proba(X_test)
```

For binær klassifikasjon kan man hente sannsynligheten for klasse 1 slik:

```python
positive_probabilities = probabilities[:, 1]
```

Dette er nyttig hvis man vil justere terskelen for klassifikasjon.

---

## Evaluere modellen

```python
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))
```

`classification_report` viser blant annet:

* precision
* recall
* f1-score
* support

---

## Confusion matrix

```python
cm = confusion_matrix(y_test, predictions)

print(cm)
```

Confusion matrix viser hvilke klasser modellen predikerer riktig og feil.

For en binær klassifikasjon kan den tolkes slik:

```text
True Negative   False Positive
False Negative  True Positive
```

---

## Visualisere confusion matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("Confusion Matrix")
plt.show()
```

Dette gir en mer lesbar visualisering av modellens feil og riktige prediksjoner.

---

## ROC-kurve og AUC

For binær klassifikasjon kan ROC-AUC brukes for å vurdere hvor godt modellen skiller mellom klassene.

```python
from sklearn.metrics import RocCurveDisplay, roc_auc_score

RocCurveDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("ROC Curve")
plt.show()

auc = roc_auc_score(y_test, positive_probabilities)
print("AUC:", auc)
```

AUC nær 1 betyr at modellen skiller godt mellom klassene. AUC rundt 0.5 betyr omtrent tilfeldig gjetting.

---

## Justere klassifikasjonsterskel

Standard terskel i binær klassifikasjon er ofte 0.5.

```python
custom_predictions = (positive_probabilities >= 0.5).astype(int)
```

Terskelen kan endres.

Eksempel:

```python
custom_predictions = (positive_probabilities >= 0.3).astype(int)
```

Lavere terskel gir ofte høyere recall, men kan gi flere false positives.

Høyere terskel gir ofte høyere precision, men kan gi flere false negatives.

---

## Skalering

Noen klassifikasjonsmodeller fungerer bedre når dataene skaleres.

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

Skalering er ofte viktig for:

* Logistic Regression
* KNN
* SVM
* Neural Networks

Tree-baserte modeller som Decision Tree og Random Forest trenger vanligvis ikke skalering.

---

## Pipeline

Pipeline samler preprocessing og modelltrening i én struktur.

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)
```

Pipeline gjør koden ryddigere og reduserer risikoen for feil.

---

## Eksempel med Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

Random Forest fungerer ofte godt som en sterk baseline for klassifikasjon og krever vanligvis lite preprocessing.

---

## Eksempel med KNN

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

model = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(n_neighbors=5)
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

KNN bør som regel brukes sammen med skalering fordi modellen er basert på avstand mellom observasjoner.

---

## Full enkel klassifikasjonsflyt

```python
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

df = pd.read_csv("data.csv")

X = df[["feature_1", "feature_2", "feature_3"]]
y = df["target"]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

ConfusionMatrixDisplay.from_estimator(
    model,
    X_test,
    y_test
)

plt.title("Confusion Matrix")
plt.show()
```

---