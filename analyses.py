import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm, wilcoxon
from statsmodels.stats.multitest import multipletests
from tabulate import tabulate
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from models import dumpsterfire, t_learner, t_learner_oof


# rows = 672, covariates = 25
train_data = dict(np.load("ihdp_npci_1-100.train.npz"))
test_data = dict(np.load("ihdp_npci_1-100.test.npz"))

IHDP_NAMES = [
    "bw", "b.head", "preterm", "birth.o", "nnhealth", "momage",
    "sex", "twin", "b.marr", "mom.lths", "mom.hs", "mom.scoll",
    "cig", "first", "booze", "drugs", "work.dur", "prenatal",
    "ark", "ein", "har", "mia", "pen", "tex", "was",
]

# ------------------------------------ helper defs ------------------------------------

# outputs an interval as formatted string
def interval(lo, hi):
    return f"[{lo:.3f}, {hi:.3f}]"


def monte_carlo(est):
    est = np.asarray(est, float)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return np.median(est), lo, hi


def n_sig(pvals, alpha=0.05):
    pvals = np.asarray(pvals, float)
    return f"{int(np.sum(pvals < alpha))}/{len(pvals)}"


def hist_grid(title, labels, series, filename, cols=3):
    rows = int(np.ceil(len(labels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.8 * rows), squeeze=False)

    for ax, label, v in zip(axes.flat, labels, series):
        ax.hist(v, bins=25, color="steelblue")
        ax.axvline(np.median(v), color="black")
        ax.axvline(0, color="crimson", linestyle="--")
        ax.set_title(f"{label}   median {np.median(v):.2f}", fontsize=9)

    for ax in axes.flat[len(labels):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)

# ------------------------------------ analyses ------------------------------------

def model_accuracy(base, tl, cates_oracle):
    def get_pehes(cates):
        return np.array([np.sqrt(np.mean((c - o) ** 2))
                         for c, o in zip(cates, cates_oracle)])

    def stats(pehes):
        e, lo, hi = monte_carlo(pehes)
        return [e, interval(lo, hi)]

    def stats_compare(i, j):
        d = i - j
        e, lo, hi = monte_carlo(d)

        # hodges lehmann estimate
        m = len(d)
        walsh = np.sort(((d[:, None] + d) / 2)[np.triu_indices(m)])
        k = np.clip(int(m * (m + 1) / 4 - 1.96 * np.sqrt(m * (m + 1) * (2 * m + 1) / 24)),
                    0, (len(walsh) - 1) // 2)

        return [e, interval(lo, hi),
                np.median(walsh), interval(walsh[k], walsh[-k - 1]),
                wilcoxon(i, j).pvalue,
                np.median(100 * (1 - i / j))]

    b, t_l = get_pehes(base), get_pehes(tl)

    names = ["baseline", "tlearner"]
    pairs = [("tl - base", t_l, b)]

    print("\n" + tabulate([[name] + stats(pehes) for name, pehes in zip(names, (b, t_l))],
                          headers=["model", "median", "95% replication interval"],
                          floatfmt=("", ".3f", ""),
                          colalign=("left",) * 3,
                          tablefmt="github"))

    print("\n" + tabulate([[label] + stats_compare(i, j) for label, i, j in pairs],
                          headers=["comparison", "median diff", "95% replication interval", "lehmann", "95% CI", "p (wilcoxon)", "% reduction"],
                          floatfmt=("", ".3f", "", ".3f", "", ".3g", ".1f"),
                          colalign=("left",) * 7,
                          tablefmt="github"))

    hist_grid("pehe per rep", names, [b, t_l], "hist_pehe.png", cols=2)


def blp(t, y, cates, pred_t, pred_y0):
    est, pvals = [], []

    for t_i, y_i, c_i, pt_i, py0_i in zip(t, y, cates, pred_t, pred_y0):
        t_residual = t_i - pt_i
        cols = pd.DataFrame({'pred_y0': py0_i,
                             'b1': t_residual,
                             'b2': t_residual * (c_i - c_i.mean())})

        fit = sm.WLS(y_i, sm.add_constant(cols),
                     weights=1 / (pt_i * (1 - pt_i))).fit(cov_type='HC3')

        est.append(fit.params[['b1', 'b2']])
        pvals.append(fit.pvalues[['b1', 'b2']])

    est, pvals = np.array(est).T, np.array(pvals).T
    labels = ["ate (b1)", "het (b2)"]

    rows = []
    for label, e, p in zip(labels, est, pvals):
        med, lo, hi = monte_carlo(e)
        rows.append([label, med, interval(lo, hi), n_sig(p)])

    print("\n**blp**")
    print(tabulate(rows,
                   headers=["term", "median", "95% replication interval", "reps p < .05"],
                   floatfmt=("", ".3f", "", ""),
                   colalign=("left",) * 4,
                   tablefmt="github"))


def gates(t, y, cates, pred_t, pred_y0):
    gammas = [f'gamma{k+1}' for k in range(5)]
    est, pvals = [], []

    for t_i, y_i, c_i, pt_i, py0_i in zip(t, y, cates, pred_t, pred_y0):
        #replacing with ranks prevents issue where qcut fails due to same numbers
        g = pd.qcut(pd.Series(c_i).rank(method='first'), 5, labels=False).to_numpy()
        d = t_i - pt_i
        cols = pd.DataFrame({'pred_y0': py0_i,
                             **{gammas[k]: d * (g == k) for k in range(5)}})

        fit = sm.WLS(y_i, sm.add_constant(cols),
                     weights=1 / (pt_i * (1 - pt_i))).fit(cov_type='HC3')
        contrast = fit.t_test('gamma5 - gamma1')

        est.append(list(fit.params[gammas]) + [float(np.squeeze(contrast.effect))])
        pvals.append(list(fit.pvalues[gammas]) + [float(np.squeeze(contrast.pvalue))])

    est, pvals = np.array(est).T, np.array(pvals).T
    labels = [f"group {k+1}" for k in range(5)] + ["gamma5-gamma1"]

    rows = []
    for label, e, p in zip(labels, est, pvals):
        med, lo, hi = monte_carlo(e)
        rows.append([label, med, interval(lo, hi), n_sig(p)])

    print("\n**gates**")
    print(tabulate(rows,
                   headers=["group", "median", "95% replication interval", "reps p < .05"],
                   floatfmt=("", ".3f", "", ""),
                   colalign=("left",) * 4,
                   tablefmt="github"))

    hist_grid("gates per rep", labels, est, "hist_gates.png")


def gates_oracle(cates_oracle):
    est = []

    for c in cates_oracle:
        g = pd.qcut(pd.Series(c).rank(method='first'), 5, labels=False).to_numpy()
        est.append([c[g == k].mean() for k in range(5)])

    est = np.array(est).T
    labels = [f"group {k+1}" for k in range(5)]

    rows = []
    for label, e in zip(labels, est):
        med, lo, hi = monte_carlo(e)
        rows.append([label, med, interval(lo, hi)])

    print("\n**gates oracle**")
    print(tabulate(rows,
                   headers=["group", "median", "95% replication interval"],
                   floatfmt=("", ".3f", ""),
                   colalign=("left",) * 3,
                   tablefmt="github"))


def clan(x, cates, names=IHDP_NAMES):
    diffs, pvals = [], []

    for x_i, c_i in zip(x, cates):
        g = pd.qcut(pd.Series(c_i).rank(method='first'), 5, labels=False).to_numpy()
        low, high = x_i[g == 0], x_i[g == 4]

        diff = high.mean(0) - low.mean(0)
        se = np.sqrt(low.var(0, ddof=1) / len(low) + high.var(0, ddof=1) / len(high))
        p = 2 * norm.sf(np.abs(diff / np.maximum(se, 1e-12)))

        diffs.append(diff)
        pvals.append(multipletests(p, method='holm')[1])

    diffs, pvals = np.array(diffs).T, np.array(pvals).T

    rows = []
    for name, d, p in zip(names, diffs, pvals):
        med, lo, hi = monte_carlo(d)
        rows.append([name, med, interval(lo, hi), n_sig(p)])

    print("\n**clan**")
    print(tabulate(rows,
                   headers=["covariate", "median diff", "95% replication interval", "reps p adj (holm) < .05"],
                   floatfmt=("", ".3f", "", ""),
                   colalign=("left",) * 4,
                   tablefmt="github"))


def qini(t, y, cates, pred_t, pred_y0, pred_y1, cates_oracle, n_points=101):
    observed_curves, oracle_curves = [], []

    for t_i, y_i, c_i, pt_i, py0_i, py1_i, co_i in zip(t, y, cates, pred_t, pred_y0, pred_y1, cates_oracle):
        n = len(c_i)
        ks = np.round(np.linspace(0, 1, n_points) * n).astype(int)

        order = np.argsort(-c_i)
        t_o, y_o, p_o = t_i[order], y_i[order], pt_i[order]
        y0_o, y1_o = py0_i[order], py1_i[order]

        scores = (y1_o - y0_o
                  + t_o * (y_o - y1_o) / p_o
                  - (1 - t_o) * (y_o - y0_o) / (1 - p_o))
        observed_curves.append(np.concatenate([[0], np.cumsum(scores)])[ks])

        n_o = len(co_i)
        ks_o = np.round(np.linspace(0, 1, n_points) * n_o).astype(int)
        oracle_curves.append(np.concatenate([[0], np.cumsum(co_i[np.argsort(-co_i)])])[ks_o])

    observed = np.median(np.array(observed_curves), axis=0)
    oracle = np.median(np.array(oracle_curves), axis=0)
    fractions = np.linspace(0, 1, len(observed))

    area_obs = np.trapezoid(observed - fractions * observed[-1], fractions)
    area_orc = np.trapezoid(oracle - fractions * oracle[-1], fractions)

    plt.figure(figsize=(7, 5))
    plt.plot(fractions, observed, label=f"t-learner (area {area_obs:.3f})")
    plt.plot(fractions, oracle, label=f"perfect ranking (area {area_orc:.3f})", linestyle="--")
    plt.plot(fractions, fractions * observed[-1], color="gray", linewidth=1, label="random")
    plt.xlabel("fraction treated")
    plt.ylabel("cumulative gain")
    plt.legend()
    plt.tight_layout()
    plt.savefig("qini.png", dpi=150)

# ------------------------------------------------------------------------

def run_rep(i):
    t = train_data["t"][:, i]
    y = train_data["yf"][:, i]
    x = train_data["x"][:, :, i]

    t_test = test_data["t"][:, i]
    y_test = test_data["yf"][:, i]
    x_test = test_data["x"][:, :, i]

    cates_oof, pred_t_oof, pred_y0_oof, pred_y1_oof = t_learner_oof(t, y, x)

    return {
        "t": t, "y": y, "x": x,
        "cates_oracle_train": train_data["mu1"][:, i] - train_data["mu0"][:, i],
        "cates_oracle_test": test_data["mu1"][:, i] - test_data["mu0"][:, i],
        "cate_dumpsterfire": dumpsterfire(t_test, y_test),
        "cates_tl": t_learner(t, y, x, x_test)[0],
        "cates_oof": cates_oof,
        "pred_t_oof": np.clip(pred_t_oof, 0.05, 0.95),
        "pred_y0_oof": pred_y0_oof,
        "pred_y1_oof": pred_y1_oof,
    }


if __name__ == '__main__':
    reps = Parallel(n_jobs=-1, verbose=10)(delayed(run_rep)(i) for i in range(100))
    r = {k: [rep[k] for rep in reps] for k in reps[0]}

    qini(r["t"], r["y"], r["cates_oof"], r["pred_t_oof"], r["pred_y0_oof"], r["pred_y1_oof"], r["cates_oracle_train"])
    model_accuracy(r["cate_dumpsterfire"], r["cates_tl"], r["cates_oracle_test"])
    blp(r["t"], r["y"], r["cates_oof"], r["pred_t_oof"], r["pred_y0_oof"])
    gates(r["t"], r["y"], r["cates_oof"], r["pred_t_oof"], r["pred_y0_oof"])
    gates_oracle(r["cates_oracle_train"])
    clan(r["x"], r["cates_oof"])
