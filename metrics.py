import pandas as pd
import numpy as np

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    y_hat = pd.Series(y_hat)
    assert(y_hat.size == y.size)
    y_hat = list(y_hat)
    y = list(y)
    l = len(y)
    match = 0
    for j in range(l):
        if (y[j]==y_hat[j]):
            match +=1
    ans = match/l
    return ans

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    y_hat = pd.Series(y_hat)
    assert(y_hat.size == y.size)
    l = len(y)
    y_hat = list(y_hat)
    match = 0
    y = list(y)
    deno = 0
    for j in range(l):
        if (y[j]==y_hat[j]) and (y_hat[j]==cls):
            match += 1
        if (y_hat[j]==cls):
            deno += 1
    if deno==0:
        ans = 1
    else:
        ans = match/deno
    return ans

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    y_hat = pd.Series(y_hat)
    assert(y_hat.size == y.size)
    l = len(y)
    y_hat = list(y_hat)
    y = list(y)
    match = 0
    deno = 0
    for j in range(l):
        if (y[j]==y_hat[j]) and (y_hat[j]==cls):
            match += 1
        if (y[j]==cls):
            deno += 1
    if deno==0:
        ans = 1
    else:
        ans = match/deno
    return ans

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    y_hat = pd.Series(y_hat)
    assert(y_hat.size == y.size)
    sm = 0.0
    l = len(y)
    # print(l)
    for j in range(l):
        sm += (y_hat[j]-y[j])**2
        # print(sm)
    mse = sm/float(l)
    ans = np.sqrt(mse)
    # print(ans)
    return ans

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    y_hat = pd.Series(y_hat)
    assert(y_hat.size == y.size)
    sm = 0.0
    l = len(y)
    for j in range(l):
        sm += np.absolute(y_hat[j]-y[j])
    ans = sm/float(l)
    return ans
