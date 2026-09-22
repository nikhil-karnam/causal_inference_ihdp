import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.set_num_threads(1)


def dumpsterfire(t, y):
    return np.mean(y[t==1]) - np.mean(y[t==0])


class DragonNet(nn.Module):
    def __init__(self, reg_on=True):
        super().__init__()
        self.reg_on = reg_on

        self.layer1_z = nn.Linear(25, 200)
        self.layer2_z = nn.Linear(200, 200)
        self.layer3_z = nn.Linear(200, 200)

        self.output_layer_t = nn.Linear(200, 1)

        self.layer1_y0 = nn.Linear(200, 100)
        self.layer2_y0 = nn.Linear(100, 100)
        self.output_layer_y0 = nn.Linear(100, 1)

        self.layer1_y1 = nn.Linear(200, 100)
        self.layer2_y1 = nn.Linear(100, 100)
        self.output_layer_y1 = nn.Linear(100, 1)

        self.epsilon = nn.Parameter(torch.randn(1) * 0.05)

    def forward(self, x):
        z = F.elu(self.layer1_z(x))
        z = F.elu(self.layer2_z(z))
        z = F.elu(self.layer3_z(z))

        pred_t = torch.sigmoid(self.output_layer_t(z)).squeeze(-1)

        pred_y0 = F.elu(self.layer1_y0(z))
        pred_y0 = F.elu(self.layer2_y0(pred_y0))
        pred_y0 = self.output_layer_y0(pred_y0).squeeze(-1)

        pred_y1 = F.elu(self.layer1_y1(z))
        pred_y1 = F.elu(self.layer2_y1(pred_y1))
        pred_y1 = self.output_layer_y1(pred_y1).squeeze(-1)

        return pred_t, pred_y0, pred_y1

    def loss(self, t, y, pred_t, pred_y0, pred_y1):
        pred_t_safe = (pred_t + 0.01) / 1.02
        #if participant is treated, compare only y1. vice versa.
        pred_y = torch.where(t == 1, pred_y1, pred_y0)

        loss_t = F.binary_cross_entropy(pred_t_safe, t, reduction='sum')
        loss_y = torch.sum((y - pred_y) ** 2)
        #this is literally just inverse propensity weights.
        h = t / pred_t_safe - (1 - t) / (1 - pred_t_safe)
        #epsilon is just a weight itself, but not for a covariate, for the inverse propensity weight.
        #sure, we can predict y using a normal neural network. but the weight is adjusting the data
        #you predict from so that a type of covariate profile is equally represented on both sides.
        loss_reg = torch.sum((y - (pred_y + self.epsilon * h)) ** 2)

        if self.reg_on:
            return loss_t + loss_y + loss_reg
        return loss_y + loss_t

    def do_it(self, t, y, x, optimizer, epochs, patience):
        strikes = 0;
        loss_best = float('inf')

        for i in range(epochs):
            #runs randomized batches of 64 covering first 80% of rows
            order = torch.randperm(538)
            for j in range(0, 538, 64):
                batch = order[j:j + 64]
                optimizer.zero_grad()
                loss = self.loss(t[batch], y[batch], *self(x[batch]))
                loss.backward()
                optimizer.step()

            #comupte loss of validation set (last 20%)
            range_val = torch.arange(538, 672)
            loss_val = self.loss(t[range_val], y[range_val], *self(x[range_val]))

            #if loss has not improved over patience* epochs, stop early
            if loss_val < loss_best:
                state_best = {k: v.clone() for k, v in self.state_dict().items()}
                loss_best = loss_val
                strikes = 0
            else:
                strikes += 1;
                if(strikes == patience):
                    break

        self.load_state_dict(state_best)

    def fit(self, t, y, x):
        #scaling y's to mean=0, SD=1 helps training
        self.y_mean = y.mean()
        self.y_std = y.std()
        y = (y - self.y_mean) / self.y_std

        #outcome head gets decay, other heads do not
        groups = [{"params": [p for n, p in self.named_parameters() if ("_y0" in n or "_y1" in n) and n.endswith(".weight")], "weight_decay": .001},
                  {"params": [p for n, p in self.named_parameters() if not (("_y0" in n or "_y1" in n) and n.endswith(".weight"))], "weight_decay": 0}]

        optimizer = torch.optim.Adam([dict(g) for g in groups], lr=1e-3)
        self.do_it(t, y, x, optimizer, epochs=100, patience=2)

        optimizer = torch.optim.SGD([dict(g) for g in groups], lr=1e-5, momentum=0.9, nesterov=True)
        self.do_it(t, y, x, optimizer, epochs=300, patience=40)

    def predict(self, x_test):
        self.eval()
        with torch.no_grad():
            pred_t, pred_y0, pred_y1 = self(x_test)

        #unscale predicted y's
        pred_y0 = pred_y0 * self.y_std + self.y_mean
        pred_y1 = pred_y1 * self.y_std + self.y_mean
        cates = pred_y1 - pred_y0

        return cates.numpy(), pred_t.numpy(), pred_y0.numpy(), pred_y1.numpy()


def to_tensor(a):
    return torch.as_tensor(np.asarray(a).copy(), dtype=torch.float32)


def run_dragonnet(t, y, x, x_test, seed, reg):
    torch.manual_seed(seed)
    model = DragonNet(reg_on=reg)
    model.fit(to_tensor(t), to_tensor(y), to_tensor(x))
    return model.predict(to_tensor(x_test))[0]


def xlearner(t, y, x, x_test):
    #split participants into control = 0 and treated = 1
    x0 = x[t==0]
    y0 = y[t==0]
    x1 = x[t==1]
    y1 = y[t==1]

    #train outcome using covariates
    model_y0 = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
    model_y0.fit(x0, y0)
    model_y1 = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
    model_y1.fit(x1, y1)

    #d = y1 - y0 per participant. sub unknown y's with counterfactual
    d0 = model_y1.predict(x0) - y0
    d1 = y1 - model_y0.predict(x1)

    #train d using covariates
    model_d0 = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
    model_d0.fit(x0, d0)
    model_d1 = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
    model_d1.fit(x1, d1)

    #train propensity scores for all participants
    model_t = make_pipeline(StandardScaler(), LogisticRegression())
    model_t.fit(x, t)

    #run on test data
    d0_pred = model_d0.predict(x_test)
    d1_pred = model_d1.predict(x_test)
    pred_t = model_t.predict_proba(x_test)[:,1]

    cates = d0_pred * pred_t + d1_pred * (1-pred_t)

    return cates, pred_t, model_y0.predict(x_test), model_y1.predict(x_test)


#gates analysis cannot use test data because too few rows. alternative is obtaining cates on train data using cross-fitting
def xlearner_oof(t, y, x):
    cates, pred_t, pred_y0, pred_y1 = np.zeros((4, len(x)))

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for i_train, i_hold in kf.split(x):
        cates[i_hold], pred_t[i_hold], pred_y0[i_hold], pred_y1[i_hold] = xlearner(t[i_train], y[i_train], x[i_train], x[i_hold])

    return cates, pred_t, pred_y0, pred_y1
