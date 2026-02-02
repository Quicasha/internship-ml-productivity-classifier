# Occupancy Prediction from Environmental Sensor Data

This project focuses on predicting room occupancy using environmental sensor data.
The goal is to evaluate how well different machine learning models can infer human presence
based on temperature, humidity, light intensity, CO₂ concentration, and derived features.

The project is designed as a complete, end-to-end machine learning pipeline:
from data loading and preprocessing, through model training and evaluation,
to baseline comparison and cross-validation.

Special attention is paid to proper evaluation practices in order to avoid overly optimistic
results caused by data leakage or lucky train/test splits.

Given time-ordered sensor measurements, the task is to classify whether a room is occupied (1)
or not occupied (0) at a given time step.

This is a binary classification problem with temporal structure,
where model evaluation must respect the chronological nature of the data.

Accurate occupancy detection has practical applications in:
- smart building automation,
- energy efficiency optimization,
- HVAC system control,
- privacy-preserving presence detection (no cameras).

The dataset used in this project is commonly referenced in academic literature,
making it suitable both for learning purposes and realistic experimentation.

