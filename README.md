---
title: Car Price Prediction
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "3.10"
app_file: main.py
pinned: false
---

# Car Price Prediction API

This is a car price prediction model deployed on Hugging Face Spaces.

## How to use
Send a POST request to `/predict` with car features.

## Model Info
- Trained with scikit-learn
- Uses one-hot encoding
- Returns predicted price in USD
