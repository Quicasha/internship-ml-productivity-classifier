# Occupancy Prediction from Environmental Sensor Data

> End-to-end machine learning project for room occupancy detection using environmental sensor data.

---

## 📌 Project Overview

This project focuses on predicting **room occupancy** based on environmental sensor measurements.
The objective is to evaluate how well different machine learning models can infer **human presence**
from physical signals such as temperature, humidity, light intensity, CO₂ concentration, and derived features.

The project is implemented as a **complete machine learning pipeline**, covering:

- data loading and preprocessing,
- baseline modeling,
- supervised model training,
- robust evaluation and comparison,
- and analysis of model behavior.

A strong emphasis is placed on **proper evaluation practices** to avoid misleading results caused by
data leakage or overly optimistic train/test splits.

---

## 🎯 Problem Definition

Given **time-ordered sensor measurements**, the task is to classify whether a room is:

- **occupied (1)**  
- **not occupied (0)**  

at a given time step.

This is a **binary classification problem with temporal structure**, meaning that
model evaluation must respect the **chronological order of the data**.

---

## 🧠 Why This Problem Matters

Accurate occupancy detection has practical, real-world applications such as:

- 🏢 smart building automation,
- ⚡ energy efficiency optimization,
- ❄️ HVAC system control,
- 🔒 privacy-preserving presence detection (no cameras involved).

The dataset used in this project is **widely referenced in academic literature**,
making it suitable both for learning purposes and realistic experimentation.

