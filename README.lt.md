<p align="right">
<a href="README.md">🇺🇸 English</a> | 🇱🇹 Lietuvių
</p>

# Patalpų užimtumo prognozavimas pagal aplinkos jutiklių duomenis

> End-to-end mašininio mokymosi projektas, skirtas patalpų užimtumui nustatyti naudojant aplinkos jutiklių duomenis.

---

## 📌 Projekto apžvalga

Šis projektas skirtas **patalpų užimtumo (occupancy)** prognozavimui remiantis aplinkos jutiklių matavimais.
Pagrindinis tikslas – įvertinti, kaip skirtingi mašininio mokymosi modeliai geba nustatyti **žmogaus buvimą**
pagal fizinius signalus, tokius kaip temperatūra, drėgmė, apšvietimo intensyvumas, CO₂ koncentracija
ir išvestinės (derived) savybės.

Projektas įgyvendintas kaip **pilnas mašininio mokymosi pipeline**, apimantis:

- duomenų įkėlimą ir paruošimą (preprocessing);
- bazinių (baseline) modelių taikymą;
- prižiūrimą (supervised) modelių treniravimą;
- patikimą modelių vertinimą ir palyginimą;
- modelių elgsenos ir savybių analizę.

Didelis dėmesys skiriamas **teisingoms modelių vertinimo praktikoms**, siekiant išvengti klaidinančių rezultatų,
kuriuos gali sukelti duomenų nutekėjimas (data leakage) arba pernelyg optimistiniai train/test padalinimai.

---

## 🎯 Problemos apibrėžimas

Turint **laiko atžvilgiu išrikiuotus jutiklių matavimus**, užduotis yra nustatyti,
ar patalpa tam tikru laiko momentu yra:

- **užimta (1)**  
- **neužimta (0)**  

Tai yra **dvejetainės klasifikacijos uždavinys su laiko struktūra**, todėl
modelių vertinimas privalo **gerbti chronologinę duomenų tvarką**.

---

## 🧠 Kodėl ši problema svarbi

Tikslus patalpų užimtumo nustatymas turi realias praktines pritaikymo sritis, tokias kaip:

- 🏢 išmaniųjų pastatų automatizavimas;
- ⚡ energijos vartojimo efektyvumo optimizavimas;
- ❄️ HVAC sistemų valdymas;
- 🔒 privatumo nepažeidžiantis buvimo nustatymas (be kamerų).

Šiame projekte naudojamas duomenų rinkinys yra **plačiai cituojamas akademinėje literatūroje**,
todėl jis tinkamas tiek mokymosi tikslams, tiek realistiškiems eksperimentams.

---

## 📁 Projekto struktūra

```text
internship-ml-productivity-classifier/
│
├── data/
│   └── occupancy.csv
│       # Originalus, laiko atžvilgiu surikiuotas aplinkos jutiklių duomenų rinkinys
│
├── src/
│   ├── load_data.py
│   │   # Duomenų įkėlimo ir pirminio apdorojimo pagalbinės funkcijos
│   │
│   ├── preprocess.py
│   │   # Savybių (features) parinkimas ir duomenų paruošimas modeliams
│   │
│   ├── clean_data.py
│   │   # Duomenų valymas ir pasirenkamas laiko pagrindu išvestų savybių kūrimas
│   │
│   ├── metrics.py
│   │   # Centralizuotas vertinimo metrikų skaičiavimas ir standartizuotas išvedimas
│   │
│   ├── train_dummy.py
│   │   # Bazinis modelis (DummyClassifier – dažniausios klasės prognozė)
│   │
│   ├── train_logistic.py
│   │   # Logistic Regression modelis su savybių skaliavimu
│   │
│   ├── train_random_forest.py
│   │   # Random Forest klasifikatorius
│   │
│   ├── cross_validation.py
│   │   # Kryžminės validacijos logika patikimam modelių vertinimui
│   │
│   ├── compare_models.py
│   │   # Vieninga modelių palyginimo ir rezultatų agregavimo logika
│   │
│   ├── ablation_plot.py
│   │   # Savybių abliacijos analizė ir rezultatų vizualizacija
│   │
│   ├── feature_importance.py
│   │   # Random Forest savybių svarbos (feature importance) analizė
│   │
│   ├── realtime_simulation.py
│   │   # Slankiojo lango (sliding window) simuliacija, imituojanti realaus laiko prognozavimą
│   │
│   └── run.py
│       # Pagrindinis CLI įėjimo taškas modelių treniravimui, vertinimui ir palyginimui
│
├── results/
│   ├── model_comparison.csv
│   │   # Visų modelių rezultatų palyginimas vienoje lentelėje
│   │
│   ├── metrics_cv.csv
│   │   # Kryžminės validacijos apibendrintos statistikos (vidurkiai ir dispersija)
│   │
│   ├── metrics_cv_folds.csv
│   │   # Kryžminės validacijos metrikos kiekvienam atskiram fold'ui
│   │
│   ├── feature_importance.png
│   │   # Savybių svarbos vizualizacija
│   │
│   └── ablation_test.png
│       # Savybių abliacijos eksperimento rezultatų palyginimas
│
├── notebooks/
│   # Papildomi, neprivalomi eksploraciniai Jupyter notebook'ai
│
├── requirements.txt
│   # Projekto priklausomybės
│
├── .gitignore
│
└── README.md
```

---

## 🚀 Kaip paleisti projektą
Šis projektas skirtas vykdyti per **vieną CLI įėjimo tašką (`run.py`)**.  
Jokių Jupyter notebook’ų nereikia, visi rezultatai atkuriami iš komandinės eilutės.

### 1. Aplinkos paruošimas

```text
Python 3.10+
```

Rekomenduojama naudoti virtualią aplinką, kad būtų išvengta priklausomybių konfliktų:
```bash
python -m venv .venv
```

Aktvuoti virtualią aplinką:
- Windows
```bash
.venv/Scripts/activate
```

- Linux/macOS
```bash
source .venv/bin/activate
```

Įdiegti projekto priklausomybes:
```bash
pip install -r requirements.txt
```

### 2. Duomenų rinkinys

Duomenų rinkinys turi būti šioje vietoje:
```bash
data/occupancy.csv
```

Tai laiko atžvilgiu išrikiuotas aplinkos jutiklių duomenų rinkinys, turintis šiuos stulpelius:
- Temperature;
- Humidity;
- Light;
- CO2;
- HumidityRatio;
- Occupancy (tikslinė reikšmė: 0 arba 1).
 
Papildomas rankinis duomenų paruošimas prieš paleidžiant pipeline nereikalingas.


### 3. Atskirų modelių treniravimas

Visi treniravimo skriptai gali būti paleisti atskirai, tačiau rekomenduojamas būdas yra naudoti ```run.py```.

Random Forest
```bash
python src/run.py train --model rf
```

Logistic Regression (su savybių skaliavimu)
```bash
python src/run.py train --model logreg
```

Baseline (DummyClassifier – dažniausios klasės prognozė)
```bash
python src/run.py train --model dummy
```

Kiekviena komanda išveda:
- klaidų matricą (confusion matrix);
- precision / recall / F1;
- bendrą tikslumą (accuracy).


### 4. Visų modelių palyginimas (hold-out vertinimas)

Norint palyginti visus modelius tame pačiame duomenų padalinime, vykdoma:
```bash
python src/run.py compare
```

Sugeneruojamas failas:
```text
results/model_comparison.csv
```

Jame pateikiama:
- accuracy;
- class-wise precision / recall / F1;
- confusion matrix components (TN / FP / FN / TP).


### 5. Kryžminė validacija (patikimas vertinimas)

Siekiant išvengti pernelyg optimistinių rezultatų iš vieno train/test padalinimo, naudojama:
```bash
python src/run.py cross-validate
```

Sugeneruojami failai:
```text
results/metrics_cv.csv
results/metrics_cv_folds.csv
```

Šie rezultatai pateikia:
- vidurkius ir standartinius nuokrypius tarp fold’ų;
- metrikas kiekvienam atskiram fold’ui;
- įrodymą, kad rezultatai nėra atsitiktinio padalinimo pasekmė.


### 6. Savybių analizė

Savybių svarba (Random Forest)
```bash
python src/feature_importance.py
```

Rezultatas:
```text
results/feature_importance.png
```

Savybių abliacijos eksperimentas
```bash
python src/ablation_plot.py
```

Rezultatas:
```text
results/ablation_test.png
```

Šios analizės padeda suprasti, kurie jutiklių signalai turi didžiausią įtaką prognozėms.


### 7. Realiojo laiko simuliacija (nebūtina)

Norint imituoti prognozavimą realiuoju laiku, naudojant slankųjį laiko langą:
```bash
python src/realtime_simulation.py
```
Tai imituoja modelio elgseną streaming / deployment tipo scenarijuje.


### 8. Visų rezultatų atkūrimas

Minimalus pilnas paleidimo scenarijus:
```bash
pip install -r requirements.txt
python src/run.py train --model rf
python src/run.py train --model logreg
python src/run.py train --model dummy
python src/run.py compare
python src/run.py cross-validate
```
Visi rezultatai išsaugomi kataloge results/.

