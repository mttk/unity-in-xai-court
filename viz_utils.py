import os
import itertools

import pickle
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from al.sampler_mapping import AL_SAMPLERS


def load_results(base_dir="results/", dataset="IMDB", model="JWA"):
    experiments = {}

    file_names = tuple(
        f"{dataset}-{model}-{sampler}-all" for sampler in AL_SAMPLERS.keys()
    )
    for filename in os.listdir(base_dir):
        if filename.startswith(file_names) and filename.endswith(".pkl"):
            with open(os.path.join(base_dir, filename), "rb") as f:
                results, meta = pickle.load(f)
                experiments[meta["al_sampler"]] = results

    meta["interpret_pairs"] = list(itertools.combinations(meta["interpreters"], 2))
    del meta["al_sampler"]

    return experiments, meta


def cartography_average(new_df_crt):
    grouped = new_df_crt.groupby(["sampler", "al_iter"])
    new_df_crt_avg = pd.DataFrame()
    new_df_crt_avg["correctness"] = grouped.correctness.agg(np.stack).apply(
        lambda x: np.median(x, 0)
    )
    new_df_crt_avg["confidence"] = grouped.confidence.apply(np.mean)
    new_df_crt_avg["variability"] = grouped.variability.apply(np.mean)
    new_df_crt_avg["forgetfulness"] = grouped.forgetfulness.agg(np.stack).apply(
        lambda x: np.median(x, 0)
    )
    new_df_crt_avg["threshold_closeness"] = grouped.threshold_closeness.apply(np.mean)
    return new_df_crt_avg


def results_to_df(experiments, meta, mode="last"):
    if mode not in MODE_DICT:
        raise ValueError(
            f"Mode {mode} is not supported. Choose 'last' or 'best' epoch."
        )

    extract_fn = MODE_DICT[mode]
    dfs_tr = []
    dfs_agr = []
    dfs_crt_train = []
    dfs_crt_test = []
    dfs_attr = []
    for sampler, exp_set in experiments.items():
        df_tr, df_agr, df_crt_train, df_crt_test, df_attr = extract_fn(
            exp_set, meta["interpret_pairs"]
        )
        df_tr["sampler"] = sampler
        df_agr["sampler"] = sampler
        df_crt_train["sampler"] = sampler
        df_crt_test["sampler"] = sampler
        dfs_tr.append(df_tr)
        dfs_agr.append(df_agr)
        dfs_crt_train.append(df_crt_train)
        dfs_crt_test.append(df_crt_test)
        dfs_attr.append(df_attr)

    new_df_tr = pd.concat(dfs_tr)
    new_df_agr = pd.concat(dfs_agr)
    new_df_crt_train = pd.concat(dfs_crt_train)
    new_df_crt_test = pd.concat(dfs_crt_test)
    new_df_attr = pd.concat(dfs_attr)

    new_df_crt_avg_train = cartography_average(new_df_crt_train)
    new_df_crt_avg_test = cartography_average(new_df_crt_test)

    return new_df_tr, new_df_agr, new_df_crt_avg_train, new_df_crt_avg_test, new_df_attr


def extract_cartography(crts, exp_index, iter_vals, labeled_vals):
    correctness = []
    confidence = []
    variability = []
    forgetfulness = []
    threshold_closeness = []
    for crt in crts:
        correctness.append(crt["correctness"])
        confidence.append(crt["confidence"])
        variability.append(crt["variability"])
        forgetfulness.append(crt["forgetfulness"])
        threshold_closeness.append(crt["threshold_closeness"])

    df_crt = pd.DataFrame(
        {
            "al_iter": iter_vals,
            "labeled": labeled_vals,
            "correctness": correctness,
            "confidence": confidence,
            "variability": variability,
            "forgetfulness": forgetfulness,
            "threshold_closeness": threshold_closeness,
        }
    )
    df_crt["experiment"] = exp_index
    df_crt.set_index(["experiment", "al_iter"], inplace=True)
    return df_crt


def extract_attribution(attribution, exp_index):
    # Retrieve attributions from the last AL iter
    attribution_dict = attribution[-1][0]
    df = pd.DataFrame(attribution_dict)
    df["experiment"] = exp_index
    df["example"] = range(len(df))
    df.set_index(["experiment", "example"], inplace=True)
    return df


def extract_last_epoch(exp_set, interpret_pairs):
    dfs_tr = []
    dfs_agr = []
    dfs_crt_train = []
    dfs_crt_test = []
    dfs_attr = []
    for exp_index, experiment in enumerate(exp_set):
        train = experiment["train"]
        train_vals = [tr[-1]["loss"] for tr in train]
        test = experiment["eval"]
        test_vals = [te[-1]["accuracy"] for te in test]
        labeled_vals = experiment["labeled"]
        iter_vals = list(range(len(labeled_vals)))
        df_tr = pd.DataFrame(
            {
                "al_iter": iter_vals,
                "labeled": labeled_vals,
                "train_loss": train_vals,
                "test_accuracy": test_vals,
            }
        )
        df_tr["experiment"] = exp_index
        df_tr.set_index(["experiment", "al_iter"], inplace=True)

        agreement_vals = []
        interpret_vals = []
        correlation_vals = []
        for ip in interpret_pairs:
            for a, corr in zip(experiment["agreement"], experiment["correlation"]):
                interpret_vals.append(ip)
                agreement_vals.append(a[-1][ip])
                correlation_vals.append(np.array(corr[-1][ip]))

        df_agr = pd.DataFrame(
            {
                "al_iter": iter_vals * len(interpret_pairs),
                "labeled": labeled_vals * len(interpret_pairs),
                "agreement": agreement_vals,
                "correlation": correlation_vals,
                "interpreter": interpret_vals,
            }
        )
        df_agr["experiment"] = exp_index
        df_agr.set_index(["experiment", "al_iter", "interpreter"], inplace=True)

        df_crt_train = extract_cartography(
            experiment["cartography"]["train"], exp_index, iter_vals, labeled_vals
        )
        df_crt_test = extract_cartography(
            experiment["cartography"]["test"], exp_index, iter_vals, labeled_vals
        )

        df_attr = extract_attribution(experiment["attributions"], exp_index)

        dfs_tr.append(df_tr)
        dfs_agr.append(df_agr)
        dfs_crt_train.append(df_crt_train)
        dfs_crt_test.append(df_crt_test)
        dfs_attr.append(df_attr)

    new_df_tr = pd.concat(dfs_tr)
    new_df_agr = pd.concat(dfs_agr)
    new_df_crt_train = pd.concat(dfs_crt_train)
    new_df_crt_test = pd.concat(dfs_crt_test)
    new_df_attr = pd.concat(dfs_attr)
    return new_df_tr, new_df_agr, new_df_crt_train, new_df_crt_test, new_df_attr


def extract_best_epoch(exp_set, interpret_pairs):
    dfs_tr = []
    dfs_agr = []
    dfs_crt_train = []
    dfs_crt_test = []
    for exp_index, experiment in enumerate(exp_set):
        train = experiment["train"]
        test = experiment["eval"]
        train_vals, test_vals = [], []
        indices = []
        for tr, te in zip(train, test):
            accs = [t["accuracy"] for t in te]
            i = np.argmax(accs)
            indices.append(i)
            train_vals.append(tr[i]["loss"])
            test_vals.append(te[i]["accuracy"])
        labeled_vals = experiment["labeled"]
        iter_vals = list(range(len(labeled_vals)))
        df_tr = pd.DataFrame(
            {
                "al_iter": iter_vals,
                "labeled": labeled_vals,
                "train_loss": train_vals,
                "test_accuracy": test_vals,
            }
        )
        df_tr["experiment"] = exp_index
        df_tr.set_index(["experiment", "al_iter"], inplace=True)

        agreement_vals = []
        interpret_vals = []
        correlation_vals = []
        for ip in interpret_pairs:
            for a, corr in zip(experiment["agreement"], experiment["correlation"]):
                interpret_vals.append(ip)
                agreement_vals.append(a[-1][ip])
                correlation_vals.append(np.array(corr[-1][ip]))

        df_agr = pd.DataFrame(
            {
                "al_iter": iter_vals * len(interpret_pairs),
                "labeled": labeled_vals * len(interpret_pairs),
                "agreement": agreement_vals,
                "correlation": correlation_vals,
                "interpreter": interpret_vals,
            }
        )
        df_agr["experiment"] = exp_index
        df_agr.set_index(["experiment", "al_iter", "interpreter"], inplace=True)

        df_tr["experiment"] = exp_index
        df_tr.set_index(["experiment", "al_iter"], inplace=True)

        df_crt_train = extract_cartography(
            experiment["cartography"]["train"], exp_index, iter_vals, labeled_vals
        )
        df_crt_test = extract_cartography(
            experiment["cartography"]["test"], exp_index, iter_vals, labeled_vals
        )

        dfs_tr.append(df_tr)
        dfs_agr.append(df_agr)
        dfs_crt_train.append(df_crt_train)
        dfs_crt_test.append(df_crt_test)

    new_df_tr = pd.concat(dfs_tr)
    new_df_agr = pd.concat(dfs_agr)
    new_df_crt_train = pd.concat(dfs_crt_train)
    new_df_crt_test = pd.concat(dfs_crt_test)
    return new_df_tr, new_df_agr, new_df_crt_train, new_df_crt_test


def df_train_average(df, groupby=["al_iter", "sampler"]):
    new_df = df.groupby(groupby).aggregate("mean")
    new_df.labeled = new_df.labeled.astype(int)
    return new_df


def df_attr_average(df, groupby="example"):
    new_df = df.groupby(groupby).aggregate("mean")
    return new_df


def df_agr_average(df, groupby=["al_iter", "sampler", "interpreter"]):
    grouped = df.groupby(groupby)
    new_df = pd.DataFrame()
    new_df["correlation"] = grouped.correlation.apply(np.mean)
    new_df["aggrement"] = grouped.agreement.agg("mean")
    new_df["labeled"] = grouped.labeled.agg("min")
    return new_df


def plot_al_accuracy(data, figsize=(12, 8), ci=90):
    plt.figure(figsize=figsize)
    sns.lineplot(
        data=data,
        x="labeled",
        y="test_accuracy",
        hue="sampler",
        style="sampler",
        markers=True,
        dashes=False,
        ci=ci,
    )


def plot_experiment_set(df_tr, df_agr, meta, sampler, figsize=(12, 16)):
    df_tr_filt = df_tr[df_tr.sampler == sampler]
    df_agr_filt = df_agr[df_agr.sampler == sampler]
    _, axs = plt.subplots(3, figsize=figsize, sharex=True)
    axs[0].set_title(f"{meta['dataset']} - {meta['model']} - {sampler}")
    sns.lineplot(ax=axs[0], data=df_tr_filt, x="labeled", y="train_loss", color="r")
    sns.lineplot(ax=axs[1], data=df_tr_filt, x="labeled", y="test_accuracy", color="g")
    g = sns.lineplot(
        ax=axs[2],
        data=df_agr_filt,
        x="labeled",
        y="agreement",
        hue="interpreter",
        style="interpreter",
        markers=True,
        dashes=False,
    )
    g.legend(loc="center right", bbox_to_anchor=(1.3, 0.5), ncol=1)
    plt.show()


def scatter_it(df, meta, hue_metric="correct", show_hist=True):
    # Subsample data to plot, so the plot is not too busy.
    dataframe = df
    #     dataframe.sample(
    #         n=25000 if dataframe.shape[0] > 25000 else len(dataframe)
    #     )

    if hue_metric == "correct":
        # Normalize correctness to a value between 0 and 1.
        dataframe = dataframe.assign(
            corr_frac=lambda d: d.correctness / d.correctness.max()
        )
        dataframe = dataframe.sort_values("corr_frac")
        dataframe[hue_metric] = [f"{x:.1f}" for x in dataframe["corr_frac"]]
    elif hue_metric == "forget":
        # Normalize forgetfulness to a value between 0 and 1.
        dataframe = dataframe.assign(
            forg_frac=lambda d: d.forgetfulness / d.forgetfulness.max()
        )
        dataframe = dataframe.sort_values("forg_frac")
        dataframe[hue_metric] = [f"{x:.1f}" for x in dataframe["forg_frac"]]
    else:
        raise ValueError(
            f"Hue metric {hue_metric} is not supported. Choose from ['correct', 'forget']."
        )

    main_metric = "variability"
    other_metric = "confidence"

    hue = hue_metric
    num_hues = len(dataframe[hue].unique().tolist())
    style = hue_metric if num_hues < 8 else None

    if not show_hist:
        fig, axs = plt.subplots(1, 1, figsize=(8, 4))
        ax0 = axs
    else:
        fig = plt.figure(
            figsize=(16, 10),
        )
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 1])

        ax0 = fig.add_subplot(gs[0, :])

    ### Make the scatterplot.

    # Choose a palette.
    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(
        x=main_metric,
        y=other_metric,
        ax=ax0,
        data=dataframe,
        hue=hue,
        palette=pal,
        style=style,
        s=30,
    )

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    an1 = ax0.annotate(
        "ambiguous",
        xy=(0.9, 0.5),
        xycoords="axes fraction",
        fontsize=15,
        color="black",
        va="center",
        ha="center",
        bbox=bb("black"),
    )
    an2 = ax0.annotate(
        "easy-to-learn",
        xy=(0.27, 0.85),
        xycoords="axes fraction",
        fontsize=15,
        color="black",
        va="center",
        ha="center",
        bbox=bb("r"),
    )
    an3 = ax0.annotate(
        "hard-to-learn",
        xy=(0.35, 0.25),
        xycoords="axes fraction",
        fontsize=15,
        color="black",
        va="center",
        ha="center",
        bbox=bb("b"),
    )

    if not show_hist:
        plot.legend(
            ncol=1,
            bbox_to_anchor=(1.01, 0.5),
            loc="center left",
            fancybox=True,
            shadow=True,
        )
    else:
        plot.legend(fancybox=True, shadow=True, ncol=1)
    plot.set_xlabel("variability")
    plot.set_ylabel("confidence")

    if show_hist:
        plot.set_title(
            f"{meta['dataset']} Data Map - {meta['model']} model - {len(df)} datapoints",
            fontsize=17,
        )

        # Make the histograms.
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[1, 2])

        plott0 = dataframe.hist(column=["confidence"], ax=ax1, color="#622a87")
        plott0[0].set_title("")
        plott0[0].set_xlabel("confidence")
        plott0[0].set_ylabel("density")

        plott1 = dataframe.hist(column=["variability"], ax=ax2, color="teal")
        plott1[0].set_title("")
        plott1[0].set_xlabel("variability")

        if hue_metric == "correct":
            plot2 = sns.countplot(x="correct", data=dataframe, color="#86bf91", ax=ax3)
            ax3.xaxis.grid(True)  # Show the vertical gridlines

            plot2.set_title("")
            plot2.set_xlabel("correctness")
            plot2.set_ylabel("")

        else:
            plot2 = sns.countplot(x="forget", data=dataframe, color="#86bf91", ax=ax3)
            ax3.xaxis.grid(True)  # Show the vertical gridlines

            plot2.set_title("")
            plot2.set_xlabel("forgetfulness")
            plot2.set_ylabel("")

    fig.tight_layout()
    return fig


def plot_cartography(
    df_crt, sampler, al_iter, meta, hue_metric="correct", show_hist=True
):
    df = convert_cartography_df(df_crt, sampler, al_iter)
    return scatter_it(df, meta, hue_metric=hue_metric, show_hist=show_hist)


def convert_cartography_df(df, sampler, al_iter):
    df_i = df.loc[(sampler, al_iter)]
    new_df = pd.DataFrame(
        {
            "correctness": df_i.correctness.tolist(),
            "confidence": df_i.confidence.tolist(),
            "variability": df_i.variability.tolist(),
            "forgetfulness": df_i.forgetfulness.tolist(),
            "threshold_closeness": df_i.threshold_closeness.tolist(),
        }
    )

    return new_df


MODE_DICT = {"last": extract_last_epoch, "best": extract_best_epoch}

SAMPLERS = [
    "random",
    "margin",
    "entropy",
    "badge",
    "margin_dropout",
    "entropy_dropout",
    "batch_bald",
    "anti_entropy",
    "core_set",
    "kmeans",
]
