# SWARM Arrhythmia

A Particle Swarm Optimization (PSO) training script for a neural network on the MIT-BIH Arrhythmia dataset.

## Overview

This project uses a simple Multi-Layer Perceptron (MLP) to classify ECG heartbeats as either:
- `0` = Normal : MIT-BIH Normal (N)
- `1` = Abnormal : MIT-BIH Ventricular Ectopic (V), Supraventricular (S), Fusion (F), etc.

The network weights are optimized using PSO rather than gradient-based training. The script supports loading data either from a cached Excel file or from the PhysioNet MIT-BIH database.

## Files

- `SWARM-Arrhythmia.py` — main training and evaluation script
- `SWARM-Arrhythmia-v0.py` — earlier version of the script
- `mitbih_data.xlsx` — cached feature/label data exported by the script
- `model_results.xlsx` — saved train/test predictions after script execution
- `training_output.txt` — log file capturing printed output

## Dependencies

Install the required Python libraries before running the script:

```bash
python3 -m pip install numpy pandas torch wfdb scikit-learn openpyxl
```

If you want to keep a local `requirements.txt`, add:

```text
numpy
pandas
torch
wfdb
scikit-learn
openpyxl
```

## Usage

Run the script from the project root:

```bash
python3 SWARM-Arrhythmia.py
```

### Data source options

The script uses the `data_source` variable to decide whether to load cached data or re-download from PhysioNet.

- `data_source = "excel"`: reads from `mitbih_data.xlsx` if available
- `data_source = "database"`: downloads records from MIT-BIH and generates a new Excel cache

## Output

The script prints:
- baseline train/test accuracy
- final train/test accuracy
- full train/test classification metrics

It also saves results to:
- `model_results.xlsx` — individual observations and predictions for train and test sets
- `training_output.txt` — captured console output

## Notes

- The model performs binary classification on a subset of MIT-BIH beat types.
- If `mitbih_data.xlsx` does not exist, the script downloads data from PhysioNet.
- The PSO implementation runs for a fixed number of iterations and may take time depending on the dataset size.

## Recommended Python Version

Use Python 3.9 or newer, matching the installed packages in this repository.
