# Ocean Time-Series ML

Migrating time- and frequency-domain analysis of oceanographic measurements from MATLAB to Python, then applying machine learning techniques to the same datasets.

This project builds on earlier MATLAB work analyzing internal waves and cross-shelf exchange in the Eastern Mediterranean. The goal is twofold: re-implement the original spectral and time-series analysis in Python, and extend it with machine learning methods to surface patterns in the data.

> **Original MATLAB work:** [Time-Series Analysis of Oceanographic Data](https://github.com/faurhadass/Time-Series-Analysis-Oceanographic-Data)

## Motivation

The original analysis studied how slope geometry, model forcing, and seasonal cycles shape internal wave behavior, using time-series and spectral analysis (PSD, spectrograms, wavelets) on currents, temperature, and salinity data. Porting this to Python opens the door to the broader scientific-Python and machine-learning ecosystem, and serves as a hands-on application of ML techniques.

## Methods

- **Time-series analysis** — trends, variability, and seasonal cycles in temperature, salinity, and currents
- **Spectral analysis** — power spectral density, spectrograms, and wavelet analysis to study internal wave energy
- **Machine learning**  —  modeling across oceanographic variables

## Project Structure

```
ocean-timeseries-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── data/              # not tracked in git
│   ├── raw/           # original measurements
│   └── processed/     # cleaned / derived data
├── notebooks/         # exploratory analysis and figures
├── src/               # reusable analysis code
└── results/
    └── figures/       # saved plots and outputs
```

> **Note:** the `data/` directory is intentionally excluded from version control.

## Tech Stack

Python · NumPy · pandas · SciPy · Matplotlib · scikit-learn · Jupyter

## Status

Active and ongoing. Spectral and time-series analysis is being ported from MATLAB; machine learning components are in development.

## Author

**Hadassah Brenner-Faur**
[LinkedIn](https://linkedin.com/in/hadassah-faur) · [GitHub](https://github.com/faurhadass)
