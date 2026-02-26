import pandas as pd
from sklearn import clone
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt
import matplotlib as mpl
from mlxtend.evaluate.time_series import (
    GroupTimeSeriesSplit,
    plot_splits,
)

# Load data (add your own file path)
df = pd.read_csv(
    "/Users/kayttaja/Desktop/PEER BENCHMARKING SEMICONDUCTOR SECTOR/data/semiconductor_companies_10k_cleaned.csv"
)
df = df.sort_values(["year", "company_name"])
print(df.columns)
df = df.dropna()
print(df.isna().sum())

# Columns for training and testing
X = ["capex_margin_w", "ebit_margin_w", "revenue_growth_w"]
y = "target_next_year"

# Set up logistic regression
pipe = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ]
)

# Parameter grid for L2 regularization
param_grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0]}

# Create a list for the testing years
years = df["year"].unique()
print(years)
test_years = years[years > 2021]
test_years

# Dataframe to store the results
df_results = df.loc[df["year"] > 2021, ["year", "company_name"]].copy()
cv_monitor = []

# Set the parameters for the cv split
cv_args = {"test_size": 1, "n_splits": 5, "window_type": "expanding"}

for year in test_years:
    # Split the training and testing sets
    df_test = df[df["year"] == year]
    df_train = df[df["year"] < year]
    # Get groups
    groups = df_train["year"].to_numpy()
    # Split features and target
    X_train = df_train[X]
    y_train = df_train[y]
    X_test = df_test[X]
    y_test = df_test[y]

    # Define the group time series split
    gts = GroupTimeSeriesSplit(**cv_args)

    # Set up grid search
    GS = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=gts,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    # Fit
    GS.fit(X_train, y_train, groups=groups)

    # Extract and store cross-validation results
    res = pd.DataFrame(GS.cv_results_)
    best_row = res.loc[res["rank_test_score"].idxmin()]
    # Store the cross-validation monitoring info
    cv_monitor.append(
        {
            "Date": year,
            "Best_Params": GS.best_params_,
            "Mean_Train_Score": float(best_row["mean_train_score"]),
            "Mean_Valid_Score": float(best_row["mean_test_score"]),
        }
    )

    # Probability for positive class
    p_pos = GS.predict_proba(X_test)
    y_pred = GS.predict(X_test)

    # Store results
    idx = df_test.index
    df_results.loc[idx, "Prediction"] = y_pred
    df_results.loc[idx, "Actual"] = y_test.values
    df_results.loc[idx, "Prob_1"] = p_pos[:, 1]
    df_results.loc[idx, "Best_Params"] = str(GS.best_params_["clf__C"])


print(df_results.head())

mpl.rcParams.update({"font.size": 15})

# Confusion matrix
cm = confusion_matrix(df_results["Actual"], df_results["Prediction"])
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Logistic Regression Classifier")
plt.show()

# Compute testing precision and F1
precision = precision_score(df_results["Actual"], df_results["Prediction"])
recall = recall_score(df_results["Actual"], df_results["Prediction"])
f1 = 2 * (precision * recall) / (precision + recall + 1e-12)
print(f"Testing Precision: {precision:.4f}")
print(f"Testing Recall: {recall:.4f}")
print(f"Testing F1 Score: {f1:.4f}")

# ROC-AUC
y_true = df_results["Actual"].astype(int).to_numpy()
y_score = df_results["Prob_1"].astype(float).to_numpy()

fpr, tpr, thresholds = roc_curve(y_true, y_score)
auc = roc_auc_score(y_true, y_score)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC curve for Logistic Regression")
plt.legend()
plt.grid(True)
plt.show()

# PR-Recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_score)
ap = average_precision_score(y_true, y_score)

plt.figure()
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision–Recall curve Logistic Regression (AP={ap:.3f})")
plt.grid(True)
plt.show()

# Plot training vs validation scores
cv_df = pd.DataFrame(cv_monitor)
plt.figure(figsize=(10, 5))
plt.plot(cv_df["Date"], cv_df["Mean_Train_Score"], label="Training Score")
plt.plot(cv_df["Date"], cv_df["Mean_Valid_Score"], label="Validation Score")
plt.xlabel("Year")
plt.ylabel("Score")
plt.title("Training vs Validation Scores Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot the initial CV
df_train = df[df["year"] < 2022].copy()

X_train = df_train[X]
y_train = df_train[y]
groups = df_train["year"].to_numpy()

plot_splits(X_train, y_train, groups, **cv_args)
plt.show()

# Plot the train-test folds
start_year = 2008
first_test_year = 2022
last_test_year = 2024
years = np.arange(start_year, last_test_year + 1)
test_years = np.arange(first_test_year, last_test_year + 1)

# Colors
train_color = "tab:blue"
test_color = "tab:orange"

fig, ax = plt.subplots(figsize=(12, 3.5))

for i, ty in enumerate(test_years):
    train_mask = years < ty
    test_mask = years == ty

    ax.barh(
        i,
        train_mask.sum(),
        left=years[0],
        height=0.6,
        color=train_color,
        label="Train" if i == 0 else "",
    )
    ax.barh(
        i,
        test_mask.sum(),
        left=ty,
        height=0.6,
        color=test_color,
        label="Test" if i == 0 else "",
    )

ax.set_yticks(range(len(test_years)))
ax.set_yticklabels([str(y) for y in test_years])
ax.set_xlabel("Year")
ax.set_ylabel("Test year")
ax.set_title("Out-of-sample backtest with an expanding training window")
ax.set_xticks(years)

ax.grid(False)

ax.legend()
plt.tight_layout()
plt.show()


#########################################
# Let's perform a back-test with a threshold tuner
for year in test_years:
    # Split the training and testing sets
    df_test = df[df["year"] == year]
    df_train = df[df["year"] < year]
    # Get groups
    groups = df_train["year"].to_numpy()
    # Split features and target
    X_train = df_train[X]
    y_train = df_train[y]
    X_test = df_test[X]
    y_test = df_test[y]

    # Define the group time series split
    gts = GroupTimeSeriesSplit(**cv_args)

    # Set up grid search
    GS = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="neg_log_loss",
        cv=gts,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    # Fit
    GS.fit(X_train, y_train, groups=groups)

    # Extract and store cross-validation results
    res = pd.DataFrame(GS.cv_results_)
    best_row = res.loc[res["rank_test_score"].idxmin()]
    # Store the cross-validation monitoring info
    cv_monitor.append(
        {
            "Date": year,
            "Best_Params": GS.best_params_,
            "Mean_Train_Score": float(best_row["mean_train_score"]),
            "Mean_Valid_Score": float(best_row["mean_test_score"]),
        }
    )
    # Threshold tuner
    # Pick the chosen hyperparameters
    best_est = GS.best_estimator_

    # Create lists to store validation indices and probabilities
    valid_idx = []
    valid_proba = []

    # New splitter object for threshold tuning
    gts_valid = GroupTimeSeriesSplit(**cv_args)

    # Let's iterate through training and testing sets
    for tr_idx, va_idx in gts_valid.split(X_train, y_train, groups=groups):
        est = clone(best_est)
        # Fit the model to the training data
        est.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
        # Get the validation fold's predicted probabilities
        p_va = est.predict_proba(X_train.iloc[va_idx])[:, 1]

        # Gather the validation indices and probabilities
        valid_idx.extend(va_idx)
        valid_proba.extend(p_va)

    # Extract the validation results
    valid_idx = np.array(valid_idx)
    valid_proba = np.array(valid_proba)

    # Extract the validation data
    y_valid = y_train.iloc[valid_idx].to_numpy()

    # Get the precision and recall values at different thresholds
    precision, recall, thresholds = precision_recall_curve(y_valid, valid_proba)

    # Compute f1
    f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    # Get the index of the best F1 score
    best_i = np.argmax(f1)

    # Get the threshold for the best F1 score
    best_threshold = thresholds[best_i]
    best_f1 = f1[best_i]

    # Predict with the best threshold and estimated hyperparameters
    p_pos = best_est.predict_proba(X_test)[:, 1]
    y_pred = (p_pos >= best_threshold).astype(int)

    # Store results
    idx = df_test.index
    df_results.loc[idx, "Prediction"] = y_pred
    df_results.loc[idx, "Actual"] = y_test.values
    df_results.loc[idx, "Prob_1"] = p_pos
    df_results.loc[idx, "Best_Params"] = str(GS.best_params_["clf__C"])
    df_results.loc[idx, "Best_Threshold"] = best_threshold
    df_results.loc[idx, "Best_CV_F1"] = best_f1

print(df_results[["Best_CV_F1", "Best_Threshold"]].describe())

# Confusion matrix
cm = confusion_matrix(df_results["Actual"], df_results["Prediction"])
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for Logistic Regression Classifier")
plt.show()

# Compute testing precision and F1
precision = precision_score(df_results["Actual"], df_results["Prediction"])
recall = recall_score(df_results["Actual"], df_results["Prediction"])
f1 = 2 * (precision * recall) / (precision + recall)
print(f"Testing Precision: {precision:.4f}")
print(f"Testing Recall: {recall:.4f}")
print(f"Testing F1 Score: {f1:.4f}")
