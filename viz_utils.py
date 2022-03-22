from argparse import ArgumentError
import os
import itertools

import pickle
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr


def load_results(base_dir="results/", dataset="IMDB", model="JWA"):
    file_name = f"{dataset}-{model}"
    found = False

    for filename in os.listdir(base_dir):
        if filename.startswith(file_name) and filename.endswith(".pkl"):
            with open(os.path.join(base_dir, filename), "rb") as f:
                results, meta = pickle.load(f)
                found = True
                break

    if not found:
        raise ArgumentError(f"Result file was not found.")

    meta["interpret_pairs"] = list(itertools.combinations(meta["interpreters"], 2))

    return results, meta


def results_to_df(experiments, meta):
    df_tr, df_agr, df_crt_train, df_crt_test, df_attr = extract_data(
        experiments, meta["interpret_pairs"]
    )

    return (
        df_tr,
        df_agr,
        df_crt_train,
        df_crt_test,
        df_attr,
    )


def extract_data(exp_set, interpret_pairs):
    dfs_tr = []
    dfs_agr = []
    dfs_crt_train = []
    dfs_crt_test = []
    dfs_attr = []
    for exp_index, experiment in enumerate(exp_set):
        train = experiment["train"]
        train_vals = [tr["loss"] for tr in train]
        test = experiment["eval"]
        test_vals = [te["accuracy"] for te in test]
        df_tr = pd.DataFrame(
            {
                "epoch": range(len(train_vals)),
                "train_loss": train_vals,
                "test_accuracy": test_vals,
            }
        )
        df_tr["experiment"] = exp_index
        df_tr.set_index(["experiment", "epoch"], inplace=True)

        agreement_vals = []
        interpret_vals = []
        correlation_vals = []
        for ip in interpret_pairs:
            for a, corr in zip(experiment["agreement"], experiment["correlation"]):
                interpret_vals.append(ip)
                agreement_vals.append(a[ip])
                correlation_vals.append(np.array(corr[ip]))

        df_agr = pd.DataFrame(
            {
                "epoch": list(range(len(train_vals))) * len(interpret_pairs),
                "agreement": agreement_vals,
                "correlation": correlation_vals,
                "interpreter": interpret_vals,
            }
        )
        df_agr["experiment"] = exp_index
        df_agr.set_index(["experiment", "epoch", "interpreter"], inplace=True)

        df_crt_train = extract_cartography(
            experiment["cartography"]["train"], exp_index
        )
        df_crt_test = extract_cartography(experiment["cartography"]["test"], exp_index)

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

    return (
        new_df_tr,
        new_df_agr,
        new_df_crt_train,
        new_df_crt_test,
        new_df_attr,
    )


def extract_cartography(crt, exp_index):
    df_crt = pd.DataFrame(crt)
    df_crt["experiment"] = exp_index
    df_crt["example"] = range(len(df_crt))
    df_crt.set_index(["experiment", "example"], inplace=True)
    return df_crt


def extract_attribution(attributions, exp_index):
    dfs = []
    for i, att_dict in enumerate(attributions):
        df = pd.DataFrame(att_dict)
        df["epoch"] = i
        df["example"] = range(len(df))
        dfs.append(df)
    con_df = pd.concat(dfs)
    con_df["experiment"] = exp_index
    con_df.set_index(["experiment", "epoch", "example"], inplace=True)
    return con_df


def cartography_average(df):
    grouped = df.groupby(level=1)
    new_df = pd.DataFrame()
    new_df["correctness"] = grouped.correctness.agg(np.median)
    new_df["confidence"] = grouped.confidence.agg(np.mean)
    new_df["variability"] = grouped.variability.agg(np.mean)
    new_df["forgetfulness"] = grouped.forgetfulness.agg(np.median)
    new_df["threshold_closeness"] = grouped.threshold_closeness.agg(np.mean)
    return new_df


def agreement_average(df):
    grouped = df.groupby(level=[1, 2])
    new_df = pd.DataFrame()
    new_df["agreement"] = grouped.agreement.agg(np.mean)
    new_df["correlation"] = grouped.correlation.agg(np.mean)
    return new_df


def attribution_average(df):
    grouped = df.groupby(level=[1, 2])
    new_df = grouped.agg(np.mean)
    return new_df


# def df_train_average(df, groupby=["al_iter", "sampler"]):
#     new_df = df.groupby(groupby).aggregate("mean")
#     new_df.labeled = new_df.labeled.astype(int)
#     return new_df


# def df_attr_average(df, groupby="example"):
#     new_df = df.groupby(groupby).aggregate("mean")
#     return new_df


# def df_agr_average(df, groupby=["al_iter", "sampler", "interpreter"]):
#     grouped = df.groupby(groupby)
#     new_df = pd.DataFrame()
#     new_df["correlation"] = grouped.correlation.apply(np.mean)
#     new_df["aggrement"] = grouped.agreement.agg("mean")
#     new_df["labeled"] = grouped.labeled.agg("min")
#     return new_df


def plot_experiment(df_tr, df_agr, meta, figsize=(12, 16)):
    _, axs = plt.subplots(3, figsize=figsize, sharex=True)
    axs[0].set_title(f"{meta['dataset']} - {meta['model']}")
    sns.lineplot(ax=axs[0], data=df_tr, x="epoch", y="train_loss", color="r", ci="sd")
    sns.lineplot(
        ax=axs[1], data=df_tr, x="epoch", y="test_accuracy", color="g", ci="sd"
    )
    g = sns.lineplot(
        ax=axs[2],
        data=df_agr,
        x="epoch",
        y="agreement",
        hue="interpreter",
        style="interpreter",
        ci="sd",
        markers=True,
        dashes=True,
    )
    g.legend(loc="center right", bbox_to_anchor=(1.3, 0.5), ncol=1)
    g.set(xticks=range(meta["epochs_per_train"]))
    plt.show()


def plot_cartography(df, meta, hue_metric="correct", show_hist=True):
    dataframe = cartography_average(df)
    # Subsample data to plot, so the plot is not too busy.
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


def plot_correlations(df_agr, df_crt, meta, figsize=(10, 18), print_flag=False):
    df_agr_avg = agreement_average(df_agr)
    df_crt_avg = cartography_average(df_crt)
    corr_vals = []
    for (epoch, ip), row in df_agr_avg.iterrows():
        if print_flag:
            print(f"Epoch {epoch}")
            print("=" * 100)
            print(f"\tInterpreter pair: {ip}")
        for key in [
            "correctness",
            "confidence",
            "variability",
            "forgetfulness",
            "threshold_closeness",
        ]:
            corr = pearsonr(row["correlation"], df_crt_avg[key])
            val = {
                "interpreter": ip,
                "epoch": epoch,
                "correlation": corr[0],
                "p-value": corr[1],
                "attribute": key,
            }
            corr_vals.append(val)
            if print_flag:
                print(f"\t\t agreement vs. {key}: {corr}")

        corr = pearsonr(row["correlation"], meta["test_lengths"])
        val = {
            "interpreter": ip,
            "epoch": epoch,
            "correlation": corr[0],
            "p-value": corr[1],
            "attribute": "length",
        }
        corr_vals.append(val)
        if print_flag:
            print(f"\t\t agreement vs. length: {corr}")
            print()

    df = pd.DataFrame(corr_vals)

    ips = meta["interpret_pairs"]
    _, axs = plt.subplots(len(ips), figsize=figsize, sharex=True)

    for i, ip in enumerate(ips):
        df_filt = df[df.interpreter == ip]
        axs[i].set_title(ip)
        g = sns.lineplot(
            ax=axs[i],
            data=df_filt,
            x="epoch",
            y="correlation",
            hue="attribute",
            style="attribute",
            markers=True,
            dashes=True,
        )
        g.axhline(0, color="gray")
        g.legend(loc="center right", bbox_to_anchor=(1.3, 0.5), ncol=1)
        g.set(xticks=range(meta["epochs_per_train"]))

    return df
