# Forecasting Electricy Markets

**Goal**: Set up a data pipeline to forecast electricity prices.

**Rules/Specifications**:
- No AI. Only use AI to help debug or as a teaching assistant (ask questions to check understanding). The code should be written and the problems solved by myself for better learning.

## Plan
- [x] Find and download public dataset containing data on price ahead and electricity generation from various sources over at least several months, preferably several years.
- [x] Write script to process downloaded data. 
  - [x] Extract relevant time series, validate them, and put processed data in parquet files.
- [x] Write script to split processed dataset into train, validation, and test datasets.
- [x] Set up version control of processed data (potentially with DVC)
  - [ ] Set up online remote repository for data. Currently only local remote.
- [x] Write script for training classic Seq2Seq model.
  - [ ] Modularize code.
    - Note: Partially modularized
- [x] Make it possible to train model with Google Colab kernel in VS code for GPU capability.
- [x] Write workflow manager script.
- [ ] Write script to calculate metrics and visualize forecasting results.
- [ ] Migrate to pytorch lightning
- [ ] Write script for training transformer model.
- [ ] Write script for training state-of-the-art forecasting models.
- [ ] Create dashboard web application to visualize results.
- [ ] Expand datasets used in forecast.

## Installation instructions

Pixi is used to mangage this repository's environment. First, click the link and follow instructions to install pixi: https://pixi.prefix.dev/latest/installation/. Then, in your terminal, run

```
pixi install
```

## Datasets
- Downloaded from [Open Power Systems Data](https://open-power-system-data.org/) on 02.11.2025.