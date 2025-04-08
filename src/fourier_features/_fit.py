import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def prep_X(X, transformations: list[callable] | None = None):
    """
    Prepares the input data by applying a series of transformations as additional features.
    This function is designed to be used with time series data, where the input data
    is a 2D array-like structure (e.g., DataFrame or NumPy array).

    Parameters:
    X (array-like): The input data to be transformed. Should be a 1D array-like structure.
    transformations (list of callable): A list of transformation functions to apply to the data.

    Returns:
    array-like: The transformed data, with additional features added. Will have the same number of rows as the input data, but with len(transformations) additional columns. If no transformations are provided, the original data is returned, but will still be reshaped to 2D.
    """
    if transformations is None:
        return X.reshape(-1, 1)

    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(X.shape) > 2:
        raise ValueError("Input data must be 1D or 2D array-like structure.")
    if X.shape[1] != 1:
        raise ValueError("Input data must be 1D array-like structure.")
    if X.shape[0] == 0:
        raise ValueError("Input data must not be empty.")
    if X.shape[1] > 1:
        raise ValueError("Input data must be 1D array-like structure.")

    for transform in transformations:
        newcol = transform(X[:, 0])
        if isinstance(newcol, pd.Series):
            newcol = newcol.to_numpy()

        # Make sure the new column is 2D
        if len(newcol.shape) == 1:
            newcol = newcol.reshape(-1, 1)
        if len(newcol.shape) > 2:
            raise ValueError(
                "Transformation must return 1D or 2D array-like structure."
            )

        X = np.concatenate((X, newcol), axis=1)
    return X


def prep_y(y, transformations: list[callable] | None = None):
    """
    Prepares the target data by applying a series of transformations.

    Parameters:
    y (array-like): The target data to be transformed.
    transformations (list of callable): A list of transformation functions to apply to the data.

    Returns:
    array-like: The transformed target data.
    """
    if transformations is None:
        return y

    return y


def fit_ts_model(X, y):
    ts = {
        "model": [],
        "train_score": [],
        "test_score": [],
        "train_idx": [],
        "test_idx": [],
        "train_pred": [],
        "test_pred": [],
        "train_X": [],
        "test_X": [],
        "train_y": [],
        "test_y": [],
        "X_transforms": [],
        "y_transforms": [],
    }

    lr = LinearRegression()

    tscv = TimeSeriesSplit(n_splits=3)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        stdX = StandardScaler()
        stdY = StandardScaler()
        # Scale the data
        X_train_scaled = stdX.fit_transform(X_train)
        X_test_scaled = stdX.transform(X_test)

        y_train_scaled = stdY.fit_transform(y_train.reshape(-1, 1))

        ts["train_X"].append(X_train)
        ts["test_X"].append(X_test)
        ts["train_y"].append(y_train)
        ts["test_y"].append(y_test)
        ts["train_idx"].append(train_index)
        ts["test_idx"].append(test_index)

        ts["X_transforms"].append(stdX)
        ts["y_transforms"].append(stdY)

        # Fit the model
        lr.fit(X_train_scaled, y_train_scaled)

        # Predict
        train_pred_scaled = lr.predict(X_train_scaled)
        y_pred_scaled = lr.predict(X_test_scaled)

        # Inverse transform the predictions
        train_pred = stdY.inverse_transform(train_pred_scaled)
        y_pred = stdY.inverse_transform(y_pred_scaled)

        ts["train_pred"].append(train_pred)
        ts["test_pred"].append(y_pred)
        ts["model"].append(lr)
        ts["train_score"].append(mean_squared_error(y_train, lr.predict(X_train_scaled)))
        ts["test_score"].append(mean_squared_error(y_test, lr.predict(X_test_scaled)))

    return ts


def plot_ts_model(ts, series, X_transforms=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["blue", "orange", "green"]

    series.plot(label="Original", color="black", ax=ax, lw=1)
    X = series.to_dataframe().iloc[:, 0].to_numpy()
    X0 = prep_X(X, transformations=X_transforms)
    idx = series.to_dataframe().index.to_numpy().astype("datetime64[M]")
    for i in range(len(ts["train_pred"])):
        color = colors[i % len(colors)]
        model = ts["model"][i]
        prediction = model.predict(X0)
        fold_numb = i + 1

        df_pred = pd.DataFrame(
            {"Month": idx, f"Fold {fold_numb}": prediction.flatten().ravel()}
        ).set_index("Month")

        df_pred.plot(
            color=color,
            ax=ax,
            lw=1,
            ls="--",
        )

    ave_train_mse = np.mean(ts["train_score"])
    ave_test_mse = np.mean(ts["test_score"])

    plt.legend()
    
    plt.title(
        f"Train and test sets with predictions\n"
        f"Train RMSE: {np.sqrt(ave_train_mse):.2f} | Test RMSE: {np.sqrt(ave_test_mse):.2f}\n"
        f"y ~ X "
    )
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.show()
