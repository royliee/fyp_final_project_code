# Bitcoin Money Laundering Detection with the Elliptic Dataset

This project evaluates supervised machine-learning methods for identifying illicit Bitcoin transactions using transaction features from the Elliptic dataset. It is a final-year project focused on comparing class-imbalance strategies, ensemble feature selection, and tree-based classifiers. The implementation and saved experiment outputs are in [`Untitled24.ipynb`](Untitled24.ipynb).

## Problem Statement

Bitcoin transactions are pseudonymous and move through a public transaction network. Detecting transactions associated with illicit activity is therefore a useful financial-crime analysis task. This project formulates detection as binary classification of labeled transactions:

- **Illicit (1):** Elliptic class `1`.
- **Licit (0):** Elliptic class `2`.
- **Unknown:** excluded from the supervised experiments.

The goal is to compare precision-oriented classifiers while accounting for the strong class imbalance and exploring whether a smaller consensus-selected feature set can retain performance.

## Dataset

The notebook uses the **Elliptic Bitcoin transaction dataset**, distributed as CSV files through the [Elliptic dataset page on Kaggle](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set). Download the dataset separately; raw data is not present in this repository.

The experiment reads these two files:

| File                        | Use                                                     |
| --------------------------- | ------------------------------------------------------- |
| `elliptic_txs_features.csv` | Transaction ID, time step, and 165 transaction features |
| `elliptic_txs_classes.csv`  | Transaction IDs and class labels                        |

After merging on transaction ID and removing unknown labels, the saved notebook output reports **46,564 labeled transactions** with **166 model inputs**: one timestamp column and 165 feature columns. The class distribution is 42,019 licit (90.24%) and 4,545 illicit (9.76%). Although the dataset is graph-based, this notebook does not load the edge list or use graph-neural-network methods; each labeled transaction is modeled from its tabular features and timestamp.

## Machine-Learning Workflow

1. **Load and label data.** Read the features and classes CSV files, merge by transaction ID, remove unknown labels, and map illicit/licit classes to 1/0.
2. **Scale features.** Apply `MinMaxScaler` to the model inputs, including the timestamp.
3. **Rank features by model agreement.** Fit Random Forest, XGBoost, and LightGBM feature-importance models. Combine their top-50 ranked feature indices and retain the 20, 30, or 40 most frequently selected indices. Compare these subsets with all 166 inputs.
4. **Split and address imbalance.** For each feature set, create an 80/20 stratified random train/test split. Apply `SMOTEENN` to the training partition only (`sampling_strategy=0.3`, `random_state=42`). Class weighting is also configured for the classifiers.
5. **Tune and fit classifiers.** Use five-fold stratified cross-validation and `GridSearchCV` with average precision as the search score. The notebook then evaluates candidate probability cutoffs (0.05, 0.10, 0.20, 0.30), refits the selected estimator on the resampled training set, and evaluates the held-out test set.
6. **Inspect results over time.** Plot test metrics and confusion matrices by feature set, calculate timestamp-level illicit F1 scores, and compare predicted and observed illicit transaction counts across the dataset's time steps.

## Models Evaluated

| Model                     | Search parameters in the notebook                                                         |
| ------------------------- | ----------------------------------------------------------------------------------------- |
| Random Forest (`sklearn`) | `max_depth` 5/8; `min_samples_split` 5/10; `min_samples_leaf` 1/5; `n_estimators` 100/200 |
| XGBoost (`xgboost`)       | `max_depth` 5/7; `min_child_weight` 3/5; `n_estimators` 100/200; `learning_rate` 0.05/0.1 |
| LightGBM (`lightgbm`)     | `max_depth` 5/7; `n_estimators` 100/200; `learning_rate` 0.05/0.1                         |

The notebook uses class-balanced weighting for Random Forest and a calculated `scale_pos_weight` for XGBoost and LightGBM. It does not include a separate dummy or logistic-regression baseline.

## Experimental Results

The table below reproduces the notebook's saved held-out test results. Inference time is the measured duration of predicting the resampled training set and test set together, not test-only latency. Values are rounded to four decimal places.

| Features  | Model         |  Precision |   Accuracy | Prediction time (s) |
| --------- | ------------- | ---------: | ---------: | ------------------: |
| All (166) | Random Forest |     0.9494 |     0.9844 |              2.1789 |
| All (166) | XGBoost       |     0.9467 |     0.9887 |              0.2016 |
| All (166) | LightGBM      | **0.9605** | **0.9901** |              0.6300 |
| Top 20    | Random Forest |     0.9139 |     0.9809 |              1.5664 |
| Top 20    | XGBoost       |     0.8824 |     0.9820 |              0.1720 |
| Top 20    | LightGBM      |     0.8967 |     0.9832 |              0.4263 |
| Top 30    | Random Forest |     0.9312 |     0.9831 |              1.5059 |
| Top 30    | XGBoost       |     0.9122 |     0.9851 |              0.1382 |
| Top 30    | LightGBM      |     0.9211 |     0.9860 |              0.5735 |
| Top 40    | Random Forest |     0.9400 |     0.9843 |              1.5101 |
| Top 40    | XGBoost       |     0.9252 |     0.9867 |              0.1450 |
| Top 40    | LightGBM      |     0.9354 |     0.9878 |              0.3898 |

The all-feature confusion matrices also support the following **derived** metrics. Precision, recall, F1, and accuracy are calculated from the saved counts at each model's reported test threshold (0.50 for Random Forest; 0.30 for XGBoost and LightGBM). The notebook does not produce an aggregate recall/F1 results table or calculate ROC-AUC.

| Model (all features) | Test threshold |  Precision | Recall |   F1-score |   Accuracy |
| -------------------- | -------------: | ---------: | -----: | ---------: | ---------: |
| Random Forest        |           0.50 |     0.9494 | 0.8878 |     0.9176 |     0.9844 |
| XGBoost              |           0.30 |     0.9467 | 0.9373 |     0.9420 |     0.9887 |
| LightGBM             |           0.30 | **0.9605** | 0.9373 | **0.9488** | **0.9901** |

**Findings.** With all features, LightGBM has the strongest saved precision, recall/F1 derived from the confusion matrix, and accuracy. XGBoost has the shortest measured prediction time in the notebook's combined train-plus-test timing. All three top-feature experiments score below the corresponding all-feature results on the listed aggregate metrics; among reduced feature sets, 40 features generally performs best. Timestamp-level illicit F1 varies substantially, especially where there are very few illicit examples, so the aggregate random-split results should not be read as proof of future-period performance.

## Evaluation and Reproducibility Notes

- **Temporal evaluation is exploratory, not a forward holdout.** The test set is a stratified random split across timestamps, and timestamp itself is a model feature. The temporal plots group this random test sample by timestamp; they do not train on earlier periods and test on later periods. Those plots also use a fixed threshold of 0.20, unlike the reported aggregate test results.
- **Preprocessing and feature selection precede the split.** The scaler and feature-importance models are fit using the full labeled dataset before the train/test split. This allows information from the eventual test records to influence the representation and selected features and can make held-out performance optimistic. For a rigorous estimate, fit all preprocessing and feature selection on training data only, and use a time-ordered holdout.
- **No ROC-AUC is reported.** The notebook's aggregate results table reports precision, accuracy, and prediction time; ROC-AUC is not computed. The saved confusion matrices enable the all-feature recall/F1 derivations shown above.
- **Checkpoint files are external.** The notebook looks for model checkpoints in the dataset directory. If they are absent, it runs the hyperparameter searches and writes checkpoint files there. Existing checkpoints should be treated as environment-specific artifacts and verified before reuse.
- **Reproduce from a clean run.** The checked-in notebook has saved outputs but its execution metadata indicates it has not been executed in the current notebook state. Results above describe those saved outputs, not a fresh run performed for this README.

## Environment and Execution

The notebook uses Python with pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, LightGBM, and Matplotlib. The saved execution log identifies Python 3.13, but exact package versions are not recorded and the repository does not provide a lockfile. Install a compatible release of each dependency in a virtual environment; the core Python modules `os`, `pickle`, `collections`, and `time` are part of the standard library.

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy scikit-learn imbalanced-learn xgboost lightgbm matplotlib
```

Download and extract the Elliptic dataset. In the first code cell of `Untitled24.ipynb`, update `data_dir` to the directory containing both required CSV files. The notebook currently contains an absolute Windows path, so this change is necessary on another machine.

Launch Jupyter from the repository root:

```bash
jupyter notebook Untitled24.ipynb
```

Run the cells in order. A fresh run can take substantially longer than opening the saved notebook because it fits three feature-selection models, runs cross-validated grid searches for each model/feature subset, and trains additional models for the diagnostic plots.

## Repository Structure

```text
.
|-- README.md          # Project overview, methodology, results, and execution guide
|-- Untitled24.ipynb   # Data preparation, training, evaluation, and visualizations
```

The dataset CSV files and generated model checkpoints are stored outside the repository and are not included in version control.

## Scope

This is an educational research project and an experimental benchmark, not a production AML screening system. The reported figures depend on the notebook's current preprocessing and random-split design. Before operational use, the pipeline needs leakage-safe preprocessing, a chronological validation strategy, calibrated threshold selection, repeatability controls, and an independent external evaluation.
