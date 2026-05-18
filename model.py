# importing libraries
import numpy as np
import pandas as pd

# function for complete ridge regression process
def regression(lr, epochs, penalty):

    # reading csv file
    df = pd.read_csv("Fish.csv")

    # removing text column
    df = df.drop(columns="Species")

    # input features
    x = df.drop(columns="Weight").values

    # output column
    y = df["Weight"].values.reshape(-1,1)

    # splitting dataset into 80% train and 20% test
    split = int(0.8 * len(x))

    x_train, x_test = x[:split], x[split:]

    y_train, y_test = y[:split], y[split:]

    # standardization
    mean = x_train.mean(axis=0)

    standard_dev = x_train.std(axis=0)

    x_train = (x_train - mean) / standard_dev

    x_test = (x_test - mean) / standard_dev

    # creating weight matrix
    w = np.zeros((x_train.shape[1],1))

    # initial bias
    b = 0

    # total rows
    n = x_train.shape[0]

    # training loop
    for i in range(epochs):

        # prediction formula
        y_cap = x_train.dot(w) + b

        # error calculation
        error = y_train - y_cap

        # mse calculation
        mse = (1/n) * np.sum(error**2)

        # derivative of weights
        dw = -(2/n) * x_train.T.dot(error)

        # derivative of bias
        db = -(2/n) * np.sum(error)

        # ridge regularization
        dw = dw + 2 * penalty * w

        # updating weights
        w = w - lr * dw

        # updating bias
        b = b - lr * db

    # prediction on test data
    y_pred = x_test.dot(w) + b

    # final mse
    final_mse = np.mean((y_test - y_pred) ** 2)

    # returning outputs
    return w, b, final_mse, y_test, y_pred
