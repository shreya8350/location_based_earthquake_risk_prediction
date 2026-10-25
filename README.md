# India Seismic Activity Estimator

Estimates the chance of at least one M5.0+ earthquake within 50 km over the next 10 years
for any location in India, using USGS earthquake records from 1970 to 2026.

**This is a statistical activity estimate, not an earthquake prediction.**

## Run locally

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    python -m streamlit run app.py

## Data sources
- USGS ComCat earthquake catalogue
- geoBoundaries ADM0 (India boundary)