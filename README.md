# Forecasting Electricity Markets

### Goal: 
Set up a data pipeline to forecast electricity prices in Europe.

**Rules/Specifications**:
- No AI. Only use AI to help debug or as a teaching assistant (ask questions to check understanding). The code should be written and the problems should be solved by me for enhanced learning.

## Plan
- [x] Find and download public dataset containing data on price ahead and electricity generation from various sources over at least several months, preferably several years.
- [x] Write script to process downloaded data. 
  - [x] Extract relevant time series, validate them, and put processed data in parquet files.
- [x] Write script to split processed dataset into train, validation, and test datasets.
- [x] Set up version control of processed data (potentially with DVC)
  - [ ] Set up online remote repository for data. Currently only local remote.
- [x] Implement classic Seq2Seq model.
- [x] Make it possible to train model with Google Colab kernel in VS code for GPU capability.
- [x] Add automatic saving of model checkpoints and syncing of saved models to Google Drive.
- [x] Write workflow manager script.
  - [x] Add argument parser to workflow manager.
- [x] Write script to calculate metrics and visualize forecasting results.
  - [x] Add to workflow manager
- [ ] Implement full test suite
  - [x] Test data cleaning
  - [x] Test splitting into train, validation, and test subsets
  - [x] Test scale data
- [ ] Migrate to pytorch lightning
- [x] Implement transformer model.
- [ ] Implement various benchmark models.
  - [ ] Moving average
  - [ ] Naive seasonal forecast
  - [ ] ARIMA
- [ ] Implement state-of-the-art forecasting models.
- [ ] Create dashboard web application to visualize results.
- [ ] Expand datasets used in forecast.

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

In the default mode, a Seq2Seq model with a gated rectified unit (GRU) as the encoder and decoder is trained. Currently implemented models: Transformer and Seq2Seq with a GRU. *To be implemented soon*: training a different model by pass the model name as an argument:

```
pixi run python run.py transformer
```

Results from run will appear as figure in `results/figures`.

## Datasets
- Downloaded from [Open Power Systems Data](https://open-power-system-data.org/) on 02.11.2025.
