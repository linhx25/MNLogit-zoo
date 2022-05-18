import numpy as np
import pandas as pd
from IPython.display import display
import src.models


def test_mnlogit(data, segment=1):

    S = segment
    config = dict(
        T=10,
        I=300,
        S=segment,
        # feat_slice=[[0],[1]]
    )
    endog = data[:, 2]
    exog = data[:, 3:]

    mnlogit = src.models.MNLogit(endog, exog, **config)
    mnlogit.fit()
    res = mnlogit.summary()
    print(f"Results for Segments: {S}")
    print(res["coef"])
    print(f"Relative size of each segment for {S} segment: ", res["prob"].round(3))
    print(res["metric"])


def test_mnlogit_withStateDependence(data, segment=1):

    S = segment
    config = dict(
        T=10,
        I=300,
        S=segment,
        # feat_slice=[[0],[1]]
    )
    endog = data[:, 2]
    exog = data[:, 3:]

    mnlogit = src.models.MNLogit(endog, exog, **config)
    mnlogit.fit()
    res = mnlogit.summary()
    print(f"Results for Segments: {S}")
    print(res["coef"])
    print(f"Relative size of each segment for {S} segment: ", res["prob"].round(3))
    print(res["metric"])


def main():
    # ============================================================================
    # Col1= Customer ID
    # Col2 = Time period
    # Col3 = choice 1 if choice brand A and 2 if chose brand B
    # Col4-5 = prices for brand A and brand B, respectively
    data = pd.read_csv("./data.txt", sep="\t", header=None)
    test_mnlogit(data.values, segment=1) # mnlogit
    test_mnlogit(data.values, segment=3) # mnlogit with latent class


    T = 10
    state = data.groupby(0).shift(1).iloc[:, 1] - 1
    state = state.fillna(0).values
    data = np.concatenate([data.values, np.eye(2)[state.astype(int)]], axis=1)
    data[slice(0, None, T), 5] = 0  # assume previous purchase for the 1st observation per customer is 0
    # ============================================================================
    # Col1= Customer ID
    # Col2 = Time period
    # Col3 = choice 1 if choice brand A and 2 if chose brand B
    # Col4-5 = prices and states for brand A
    # Col6-7 = prices and states for brand B
    data = data[:,[0,1,2,3,5,4,6]] 
    test_mnlogit_withStateDependence(data, segment=1) # mnlogit
    test_mnlogit_withStateDependence(data, segment=3) # mnlogit with latent class


if __name__ == "__main__":

    main()
    

    
