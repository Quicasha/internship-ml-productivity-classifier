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
---
