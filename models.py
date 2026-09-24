import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold


def dumpsterfire(t, y):
    return np.mean(y[t==1]) - np.mean(y[t==0])


def t_learner(t, y, x, x_test):
    #split participants into control = 0 and treated = 1
    x0 = x[t==0]
    y0 = y[t==0]
    x1 = x[t==1]
    y1 = y[t==1]

    model_y0 = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
    model_y0.fit(x0, y0)
    pred_y0 = model_y0.predict(x_test)

    model_y1 = RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=1)
    model_y1.fit(x1, y1)
    pred_y1 = model_y1.predict(x_test)

    model_t = make_pipeline(StandardScaler(), LogisticRegression())
    model_t.fit(x, t)
    pred_t = model_t.predict_proba(x_test)[:, 1]

    cates = pred_y1 - pred_y0

    return cates, pred_t, pred_y0, pred_y1


#gates analysis cannot use test data because too few rows. alternative is obtaining cates on train data using cross-fitting
def t_learner_oof(t, y, x):
    cates, pred_t, pred_y0, pred_y1 = np.zeros((4, len(x)))

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for i_train, i_hold in kf.split(x):
        cates[i_hold], pred_t[i_hold], pred_y0[i_hold], pred_y1[i_hold] = t_learner(t[i_train], y[i_train], x[i_train], x[i_hold])

    return cates, pred_t, pred_y0, pred_y1