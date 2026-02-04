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
