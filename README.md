# Forecasting Electricity Markets

### Goal: 
Set up a data pipeline to forecast electricity prices in Europe using models from [Modern Time Series Forecasting with Python](https://www.amazon.de/-/en/Modern-Time-Forecasting-Python-industry-ready-ebook/dp/B0BHJ9ZX4Q).

**Rules/Specifications**:
- **No AI**. Only use AI to help debug or as a teaching assistant (ask questions to check understanding). The code should be written and the problems should be solved by me for enhanced learning.

## Project Status
- [x] **Dataset**: Find and download public dataset containing data on price ahead and electricity generation from various sources over at least several months, preferably several years.
- [x] **Data Validation and Processing**: Write script to process downloaded data. 
  - [x] Extract relevant time series, validate them, and put processed data in parquet files.
  - [x] Write script to split processed dataset into train, validation, and test datasets.
- [x] **Data Version Control**: Set up version control of processed data with DVC
  - [ ] Set up online remote repository for data. Currently only local remote.
- [x] **Workflow**: Create workflow manager script (`run.py`)
- [x] **Models**:
  - [x] Implement classic Seq2Seq model.
  - [x] Implement transformer model.
  - [x] Implement various benchmark models. To do: Add to workflow
    - [x] Moving average
    - [x] Naive seasonal forecast
    - [x] ARIMA
- [x] **Evaluation**: Set up evaluation metrics and visualization of forecasts and save them 
- [x] **GPU capability**: Enable training of model with Google Colab kernel in VS code.
- [x] **Model storage**: Add automatic saving of model checkpoints and syncing of saved models to Google Drive.
- [x] Write script to calculate metrics and visualize forecasting results.
  - [x] Add to workflow manager
- [ ] **Testing**: Implement full test suite
  - [x] Test data cleaning
  - [x] Test splitting into train, validation, and test subsets
  - [x] Test scale data
- [ ] Migrate to pytorch lightning
- [ ] Implement state-of-the-art forecasting models.
- [ ] Create dashboard web application to visualize results.
- [ ] Expand datasets used in forecast to include more recent data and other countries. Currently only Germany. 

## Quick Start

### Installation Instructions

Pixi is used to mangage this repository's environment. Click the following link and follow instructions to install pixi: https://pixi.prefix.dev/latest/installation/. Then, in your terminal, run

```
pixi install
```

### Run Pipeline

Default mode: 

```
pixi run python run.py
```

In the default mode, a Seq2Seq model is trained with a gated rectified unit (GRU) as both the encoder and decoder. Results will appear as figure in `results/figures`.

## Datasets
- Downloaded from [Open Power Systems Data](https://open-power-system-data.org/) on 02.11.2025.
