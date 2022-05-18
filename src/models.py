import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import itertools


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)


class MNLogit(object):
    __doc__ = """
    Multinomial Logit Model

    Parameters
    ----------
    endog : array_like
        `endog` is an 1-d vector of the endogenous choice, indexed from 1.

    exog : array_like
        An intercept is not included by default and should be added by the user. 
        The data should be in the form as:
                [Nobs * [X_11, ..., X_1k, X_22, ..., X_2k, ...]]_i
        where [X_j1, ..., X_jk] is the features under the j-th choice

    I : number of customer, default = 1
    T : number of period, default = length of dataset
    S : number of consumer segments, default = 1
    feat_slice: slice of the features in the same form of the data:
                [(1,...,k), (k+1,...,2k),...]
    """
    def __init__(self, endog, exog, **kwargs):
                
        # parameters
        self.J = len(np.unique(endog))
        self.T = kwargs.get("T", len(endog))
        self.I = kwargs.get("I", 1)
        self.S = kwargs.get("S", 1)
        self.N_feat = exog.shape[1]
        self.N = exog.shape[0]
        self.k = self.N_feat // self.J

        # data
        self.X = torch.tensor(exog, dtype=torch.float32)
        self.y = torch.tensor(endog, dtype=torch.float32)
        self.Z = torch.ones((self.I, self.S), dtype=torch.float32) # latent feature

        # reshape
        self.X = self.X.reshape((self.I, self.T, -1)) # x: [N_I, N_T, N_k * N_J]
        self.y = self.y.reshape((self.I, self.T)) # y: [N_I, N_T]

        # features slice
        self.feat_slc = self._get_features_slice(kwargs)

        ## training
        self.seed = kwargs.get("seed", 42)
        self.n_epochs = kwargs.get("n_epochs", 400)
        self.lr = kwargs.get("lr", 0.001)
        self.trained = False

        # model and optimizer
        set_seed(self.seed)
        beta = nn.Parameter(torch.randn((self.S, 1 + self.k)))  # [N_segments, N_params]
        gamma_0 = nn.Parameter(torch.tensor([0.0]), requires_grad=False)  # [N_segments, 1]
        gamma = nn.Parameter(torch.randn((self.S - 1,)))
        self.opt = torch.optim.SGD([beta, gamma], lr=self.lr)
        self.model = dict(
            epoch = 0,
            beta = beta, 
            gamma_0 = gamma_0, 
            gamma = gamma
        )


    def _get_features_slice(self, kwargs):
        ## feature form: [Nobs * [X_11, ..., X_1k, X_22, ..., X_2k, ...]]_i
        ## return slice: [(1,...,k), (k+1,...,2k),...]
        if kwargs.get("feat_slice", -1) != -1:
            return kwargs["feat_slice"]
        slc = [range(it-self.k, it) for it in range(0, self.N_feat, self.k)]
        slc = slc[1:] + [slc[0]]
        return slc


    def loss_fn(self):
        beta = self.model["beta"]
        gamma = torch.cat([self.model["gamma"], self.model["gamma_0"]])
        return -self.log_likelihood(beta, gamma)
        

    def log_likelihood(self, beta, gamma):
        
        pi = (self.Z * gamma).softmax(dim=1)  # [N_I, N_S]
        ############ vectorize ############
        X, y = self.X, self.y  # x: [N_I, N_T, N_k * N_J], y: [N_I, N_T]

        xb = []
        # add constant
        for j, slc in enumerate(self.feat_slc):
            if j==len(self.feat_slc)-1: # last choice is the baseline
                xb_j = torch.cat([torch.zeros(self.I, self.T, 1), X[:, :, slc]], dim=-1)  # [N_I, N_T, k]
            else:
                xb_j = torch.cat([torch.ones(self.I, self.T, 1), X[:, :, slc]], dim=-1)  # [N_I, N_T, k]
            xb_j = xb_j @ beta.T # [N_I, N_T, N_S]
            xb.append(xb_j)

        # L_i|s
        prob = 1
        D = torch.stack(xb).exp().sum(dim=0) # denominator
        for c, xb_j in enumerate(xb):
            c = c + 1 # indexed from 1

            # [N_I, N_T, N_S] ** [N_I, N_T, 1] -> [N_I, N_T, N_S]
            prob *= (xb_j.exp() / D) ** (y == c).to(y).reshape(self.I, self.T, 1)  
        prob = torch.prod(prob, dim=1)  # [N_I, N_S]

        # L_i|s * pi_s
        res = (pi * prob).sum(dim=1).log().sum()

        return res


    def fit(self):
        # training
        likelihood = []

        for epoch in tqdm(range(self.n_epochs)):
            self.opt.zero_grad()
            with torch.autograd.set_detect_anomaly(True):
                loss = self.loss_fn()
                loss.backward()
            self.opt.step()
            
            likelihood.append(-loss.item())
            if loss.item() < -likelihood[self.model["epoch"]]:
                self.model["epoch"] = epoch

        self.train_likelihood = likelihood
        self.trained = True


    def hessian(self):
        beta = self.model["beta"]
        gamma = torch.cat([self.model["gamma"], self.model["gamma_0"]])
        # Hessian (H_{beta}, H_{b,g}, H_{g,b}, H_{gamma})
        hessian = torch.autograd.functional.hessian(
            self.log_likelihood, (beta, gamma)
        )
        H_beta = hessian[0][0].permute(
            0, 2, 1, 3
        )  # [N_S * N_X * N_S * N_X] -> [N_S * N_S * N_X * N_X]        
        H_gamma = hessian[1][1]  # [N_gamma * N_gamma]

        return dict(
            H_beta = H_beta, 
            H_gamma = H_gamma,
        )


    def summary(self):
        if not self.trained:
            self.fit()
        # fetch attributes
        H = self.hessian()
        H_beta, H_gamma = H["H_beta"], H["H_gamma"]
        beta = self.model["beta"]
        gamma, gamma_0 = self.model["gamma"], self.model["gamma_0"]

        # compute std
        std_beta = torch.stack([-1 / H_beta[s, s, :, :].diag() for s in range(self.S)])
        std_beta = std_beta.detach().numpy() ** 0.5
        std_gamma = -1 / H_gamma.diag()
        std_gamma = std_gamma.detach().numpy() ** 0.5
        
        # latent probabilty (aka. class size)
        pi = torch.cat([gamma, gamma_0]).softmax(dim=0).detach().numpy()
        beta, gamma = beta.detach().numpy(), gamma.detach().numpy()

        # AIC and BIC
        epoch = self.model["epoch"]
        self.AIC = -2 * (self.train_likelihood[epoch] + beta.shape[0] * beta.shape[1] + gamma.shape[0])
        self.BIC = -2 * (self.train_likelihood[epoch]) + (
            beta.shape[0] * beta.shape[1] + gamma.shape[0]
        ) * np.log(self.N)

        # organize results
        res_1 = pd.DataFrame(
            data=np.zeros((self.S * (1 + self.k) + self.S, 3)),
            index=[
                f"beta_{it[0]}{it[1]}" 
                for it in itertools.product(np.arange(self.S), np.arange(1 + self.k))
            ] + [f"gamma_{s}" for s in range(self.S)],
            columns=["coef.", "S.E.", "T-value"],
        )
        res_1.iloc[:, 0] = np.concatenate([beta.T.ravel(), gamma, [gamma_0]])
        res_1.iloc[:, 1] = np.concatenate([std_beta.T.ravel(), std_gamma[:-1], [np.nan]])
        res_1.iloc[:, 2] = np.concatenate(
            [(beta / std_beta).T.ravel(), gamma / std_gamma[:-1], [np.nan]]
        )

        res_2 = pd.DataFrame(
            {"Log-liklihood": [self.train_likelihood[epoch]], "AIC": self.AIC, "BIC": self.BIC},
            index=[f"N_Segment_{self.S}"],
        )

        return dict(
            coef=res_1,
            metric=res_2,
            prob=pi
        )

