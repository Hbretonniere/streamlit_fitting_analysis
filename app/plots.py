import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams as mp_param
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm

from app.utils import chose_mode, sub_cat, find_outliers, compute_error_bin

plt.gca().ticklabel_format(axis="both", style="plain", useOffset=False)


# @st.cache
def trumpet_plot(
    cat,
    codes,
    param,
    outlier_threshold,
    x_range,
    y_range,
    nb_bins,
    labels,
    freq_scat=1,
    x_axis="Truemag",
    abs=False,
):

    """
    Compute and create the Scatter plot figure for the Bias of a galaxy parameter regarding its magnitude,
    for different codes.
    In the plot, the running mean and std of each bin of magnitude is plotted in solid orange.
    Markers transparency are proportional to the density of points.
    Also plot the bias distribution in the right of the main plot.

    Parameters
    ----------
    cat : pandas dataframe
        The input catalogue for computing the biases.

    codes : list of strings
        The list of codes names you want to plot. Each code will have a different subplot.

    param : string
        The name of the parameter you want to plot. It needs to be one key of the catalogue cat.

    outlier_threshold : float
        The threshold to define what is considered as an outlier. Dashed red lines will be plotted at this
        value ( + threshold and - threshold). Points above and below the threshold are removed to plot
        the running mean and bias (orange lines)

    x_range : list of float, of len two
        The x-range for each subplot.

    y_range : list of float, of len two
        The y-range for each subplot.

    nb_bins : int
        The number of bin to compute the running mean and std.

    labels : dictionary
        The corresponding name of the codes, params etc to be plotted instead of the code syntax.
        E.g. labels{"re": "Effective Radius"}

    freq_scat : int, optional, default 10
        The fraction of object to be plotted in the scatter plot

    x_axis : string, optional, default "True magnitude"
        Define the parameter to plot the Bias against. By default, the magnitude.

    abs : boolean, optional, default: False
        If True, the absolute value of the Bias is plotted.

    Returns
    ----------
    fig : The matplotlib Figure object to be plotted in main

    """

    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True

    if len(codes) > 2:
        nb_columns = 7
        dim = (15, 15)
        legend_y = 1.4
    elif len(codes) > 1:
        nb_columns = 7
        dim = (15, 10)
        legend_y = 1.15
    else:
        nb_columns = 3
        dim = (10, 10)
        legend_y = 1.15
    fig = plt.figure(constrained_layout=True, figsize=dim)
    fig.suptitle(labels[param], color="red", y=0.95, fontsize=20)

    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.1, hspace=0.5
    )

    nb_lines = int(np.ceil(len(codes) / 2))

    gs = GridSpec(nb_lines, nb_columns, figure=fig)

    if len(cat[f"True{param}"]) > 3e3:
        alpha = 0.01
    else:
        alpha = 0.1

    line = 0
    mode = chose_mode(param)

    cmaps = {
        "SE++": plt.cm.Greys,
        "metryka": plt.cm.Reds,
        "profit": plt.cm.Blues,
        "gala": plt.cm.Greens,
        "deepleg": plt.cm.Purples,
    }
    for i, code in enumerate(codes):
        cmaps[code] = LinearSegmentedColormap.from_list(
            "mycmap", cmaps[code](np.linspace(0.4, 1, 1000))
        )
        if i % 2 == 0:
            grid_scat = fig.add_subplot(gs[line, :2])
            grid_hist = fig.add_subplot(gs[line, 2:3])
        else:
            grid_scat = fig.add_subplot(gs[line, 4:6])
            grid_hist = fig.add_subplot(gs[line, 6:7])
            line += 1
        grid_hist.set_xticks([])
        grid_scat.set_title(labels[code], fontsize=15)
        code_cat, _ = sub_cat(cat, x_axis, x_range, 0)
        mag_true = code_cat[x_axis]
        error = compute_error_bin(code_cat, code, param, mode, abs=abs)

        nbins2 = 20

        indices_out = find_outliers(code_cat, code, param, mode, y_range[1], abs=abs)
        error_dropped = error.drop(error.index[indices_out])
        mag_true_dropped = mag_true.drop(mag_true.index[indices_out])

        good_mean = np.median(error)
        bins = [200, 200]
        nans = np.argwhere(np.isnan(np.array(error)))[:, 0]

        if len(nans != 0):
            error.drop(error.index[nans], inplace=True)
            mag_true.drop(mag_true.index[nans], inplace=True)

        histo, locx, locy = np.histogram2d(
            mag_true_dropped[::freq_scat], error_dropped[::freq_scat], bins=bins
        )
        density = np.array(
            [
                histo[np.argmax(a <= locx[1:]), np.argmax(b <= locy[1:])]
                for a, b in zip(mag_true_dropped, error_dropped)
            ]
        )
        sorted_indices = density.argsort()[::-1]
        color = density[sorted_indices][::freq_scat]

        grid_scat.scatter(
            mag_true_dropped.iloc[sorted_indices][::freq_scat],
            error_dropped.iloc[sorted_indices][::freq_scat],
            c=color,
            cmap=cmaps[code],
            marker=".",
            alpha=alpha,
            rasterized=True,
        )

        grid_scat.hlines(
            0.0, x_range[0], x_range[-1], color="w", linestyle="-", linewidth=1.8
        )
        grid_scat.hlines(
            0.0, x_range[0], x_range[-1], color="#0000CD", linestyle="-", linewidth=1.3
        )

        """ Plot the running mean  """
        indices_out_thres = find_outliers(
            code_cat, code, param, mode, outlier_threshold, abs=abs
        )
        error_thres = error.drop(error.index[indices_out_thres])
        mag_true_thres = mag_true.drop(mag_true.index[indices_out_thres])

        nbins2 = nb_bins
        n1, _ = np.histogram(mag_true_thres, bins=nbins2)
        sy, _ = np.histogram(mag_true_thres, bins=nbins2, weights=error_thres)
        sy2, _ = np.histogram(
            mag_true_thres, bins=nbins2, weights=error_thres * error_thres
        )

        mean = sy / n1
        std = np.sqrt(sy2 / n1 - mean * mean)
        grid_scat.errorbar(
            (_[1:] + _[:-1]) / 2,
            mean,
            yerr=std,
            color="darkorange",
            linewidth=3,
            label="running mean with std error bars",
        )
        grid_scat.hlines(
            -outlier_threshold,
            x_range[0] - 1,
            x_range[1] + 1,
            ls="--",
            color="red",
            linewidth=0.8,
        )
        grid_scat.hlines(
            outlier_threshold,
            x_range[0] - 1,
            x_range[1] + 1,
            ls="--",
            color="red",
            linewidth=0.8,
        )
        grid_scat.set_ylim(y_range[0], y_range[1])
        grid_hist.set_ylim(y_range[0], y_range[1])
        grid_scat.tick_params(axis="both", which="major", labelsize=20)
        grid_scat.set_xlabel(labels[x_axis], fontsize=20)

        if param not in ["mag", "re", "reb", "red"]:
            grid_scat.set_ylabel(
                f"$\mathrm{{Pred_{{{param}}}}} - \mathrm{{True_{{{param}}}}} $",
                fontsize=20,
            )
            # grid_scat.set_ylabel('True - Pred', fontsize=20)
        else:
            grid_scat.set_ylabel(
                f"$\dfrac{{\mathrm{{Pred_{{{param}}}}} - \mathrm{{True_{{{param}}}}}}}{{\mathrm{{True_{{{param}}}}}}}$",
                fontsize=20,
            )
        grid_scat.tick_params(
            axis="both", which="major", labelsize=12, length=6, width=1.3
        )
        grid_scat.tick_params("both", length=3, width=1, which="minor")
        # grid_scat.grid(which='major', axis='y', linewidth=0.4, linestyle='-', color='0.75')
        grid_scat.ticklabel_format(style="sci", axis="x")
        grid_scat.hlines(
            good_mean,
            x_range[0] - 0.5,
            x_range[-1] + 0.5,
            ls="--",
            color="0.75",
            lw=2,
            alpha=1,
            label="Mean value",
        )

        """ ############### Histogram ############### """
        cmaphist = plt.cm.get_cmap(cmaps[code])
        rgba = cmaphist(0.5)
        bins_hist = np.linspace(y_range[0], y_range[1], 500)
        histo = grid_hist.hist(
            error,
            bins=bins_hist,
            color=rgba,
            orientation="horizontal",
            lw=2,
            histtype="stepfilled",
        )
        grid_hist.set_yticks([])

        grid_hist.set_xlim([0, np.max(histo[0][3:-3])])
        # grid_hist.set_ylim(y_range[0], y_range[1])
        grid_hist.hlines(
            good_mean,
            0,
            np.max(histo[0]) + np.max(histo[0]) * 0.1,
            ls="--",
            color="0.75",
            lw=2,
            alpha=1,
        )
        grid_hist.hlines(
            0,
            0,
            np.max(histo[0]) + np.max(histo[0]) * 0.1,
            color="#0000CD",
            lw=2,
            alpha=0.5,
        )
        grid_hist.hlines(
            -outlier_threshold,
            0,
            np.max(histo[0]) + np.max(histo[0]) * 0.1,
            ls="--",
            color="red",
            linewidth=0.8,
        )
        grid_hist.hlines(
            outlier_threshold,
            0,
            np.max(histo[0]) + np.max(histo[0]) * 0.1,
            ls="--",
            color="red",
            linewidth=0.8,
        )
        grid_scat.set_xlim(x_range[0] - 0.5, x_range[1] + 0.5)

        if i == 0:
            grid_scat.legend(ncol=2, loc=(0, legend_y), fontsize=17)

    return fig


def plot_error_prediction(dataset, calib_mag, params, codes, x_bins, labels):

    """
    Create the error prediction calibration figure for different codes and parameters, regarding magnitude
    bins. The last bin is always the overall score (summary of all magnitude bins)

    Parameters
    ----------
    dataset : string
        The name of the dataset, e.g. 'single_sersic'. The number of subplots and place of the bars
        change accordingly

    calib_mag : dictionary of dictionaries
        Dictionary containing for each param the score of the different codes.
        e.g. calib_mag['re']['profit'] = value

    params : list of string
        The list of parameters name you want to plot. It needs to be one key of the dictionnary.

    codes : list of string
        The list of codes' name you want to plot. It needs to be one key of the dictionnary.

    x_bins : numpy array
        The bins of magnitude for which the error calibration have been computed on. Used for the x-label.

    labels : dictionary
        The corresponding name of the codes, params etc to be plotted instead of the code syntax.
        E.g. labels{"re": "Effective Radius"}

    Returns
    ----------
    fig : The matplotlib Figure object to be plotted in main

    """

    colors = {
        "SE++": "black",
        "gala": "#2F964D",
        "profit": "#2D7DBB",
        "metryka": "#D92523",
        "deepleg": "#7262AC",
    }

    if dataset in ["single_sersic", "realistic"]:
        nb_plots = max(2, len(params))
        fig, ax = plt.subplots(1, nb_plots, figsize=(6 * nb_plots, 5))
        if nb_plots > len(params):
            ax[-1].set_visible(False)
        pad = [-0.25, 0.25, -0.75, 0.75]

    else:
        nb_lines = int(np.ceil(len(params) / 2))
        fig, ax = plt.subplots(nb_lines, 2, figsize=(10, nb_lines * 5))
        pad = [-0.5, 0, 0.5]
        ax = ax.flatten()
        if len(params) % 2 != 0:
            ax[-1].set_visible(False)

    tick_labels = []
    for i in range(len(x_bins[:-1])):
        tick_labels.append(f"[{x_bins[i]:.1f}-{x_bins[i+1]:.1f}]")
    tick_labels.append("overall")

    plt.subplots_adjust(wspace=0.4, hspace=0.45)
    bins = np.array(np.linspace(0, len(x_bins) * 3, len(x_bins)))

    for i, param in enumerate(params):
        j = 0
        for code in codes:
            if code == "deepleg":
                continue
            ax[i].bar(
                bins + pad[j],
                calib_mag[code][param],
                width=0.5,
                color=colors[code],
                label=labels[code],
            )
            j += 1
        ax[i].set_xticks(bins)
        ax[i].set_xticklabels(tick_labels, fontsize=12, rotation=45)
        ax[i].set_xlabel("True VIS mag", fontsize=15)
        ax[i].set_ylabel("Fraction of well \n calibrated objects", fontsize=15)
        ax[i].set_title(labels[param], color="red", fontsize=15)
        ax[i].axhline(0.68, ls="--", color="red")
        ax[i].text(0.1, 0.7, "0.68", color="red", size=15)
        ax[i].set_ylim([0, 0.9])
    ax[0].legend(fontsize=15, ncol=4, loc=(0, 1.2))

    return fig


def plot_score(scores, labels):

    """
    Plot the global score S for different catalogues and parameters.

    Parameters
    ----------
    scores : pandas dataframe
        The scores computed in compute_score. The columns represents the different codes,
        and the lines the parameters

    labels : dictionary
        The corresponding name of the codes, params etc to be plotted instead of the code syntax.
        E.g. labels{"re": "Effective Radius"}

    Returns
    ----------
    fig : The matplotlib Figure object to be plotted in main

    """

    sizes = {"SE++": 100, "profit": 150, "metryka": 100, "deepleg": 150, "gala": 100}

    colors = {
        "SE++": "black",
        "gala": "#2F964D",
        "profit": "#2D7DBB",
        "metryka": "#D92523",
        "deepleg": "#7262AC",
    }

    symbols = {"SE++": "o", "gala": "s", "profit": "*", "metryka": "D", "deepleg": "X"}

    fig, ax = plt.subplots(figsize=(12, 8))
    codes = scores.columns
    for code in codes:
        nb_param = len(scores[code].values)
        ax.scatter(
            np.arange(1, nb_param + 1),
            list(scores[code].values),
            label=labels[code],
            s=sizes[code],
            color=colors[code],
            marker=symbols[code],
            alpha=0.9,
        )

    ax.set_xticks(np.arange(1, nb_param + 1))
    ax.legend(fontsize=15)
    xlabels = []
    for param in scores.index.values:
        xlabels.append(labels[param])

    ax.set_xticklabels(xlabels, rotation=45)
    ax.set_ylabel(r"Global Score $\mathcal{S}$", fontsize=20)
    if max(scores.max()) > 20:
        ax.set_yscale("log")
    return fig


def summary_plot(
    summary, mag_bins, dataset, x_axis, labels, show_scores=False, bd_together=False
):
    """
    Create the Summary plot representing the different metrics (Bias, Dispersion and Outlier fraction)
    for different codes and parameters.

    Parameters
    ----------
    summary : List of dictionaries, of length 3
        The three metrics for each code and parameter, summary = [bias, dispersion, outlier_fraction]
        Each metric is a dictionary of dictionaries, e.g summary[0]['profit']['re'] = value
        The codes and parameters to plot will be automatically the one present in the dictionaries

    mag_bins : numpy array
        The bins of magnitude in which the summary has been computed on.

    dataset : string
        The name of the dataset, e.g. 'single_sersic'. The number of subplots changes accordingly

    x_axis : string
        Define the parameter to plot the metrics against. e.g. "True magnitude"

    labels : dictionary
        The corresponding name of the codes, params etc to be plotted instead of the code syntax.
        E.g. labels{"re": "Effective Radius"}

    show_scores : boolean, optional, default False
        If True, print the global score S for each parameter and code on the right of the plot, ordered per
        ranking.

    bd_together ; boolean, optional, default False
        If True, plot the disk and bulges component together in the same subplot

    Returns
    ----------
    fig : The matplotlib Figure object to be plotted in main

    """

    colors = {
        "SE++": "black",
        "gala": "#2F964D",
        "profit": "#2D7DBB",
        "metryka": "#D92523",
        "deepleg": "#7262AC",
    }

    symbols = {"SE++": "o", "gala": "s", "profit": "*", "metryka": "D", "deepleg": "X"}

    mp_param.update({"font.size": 18})
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True

    means = summary[0]
    stds = summary[1]
    outs = summary[2]
    global_scores = summary[4]
    params = list(means.keys())
    codes = list(means[list(means.keys())[0]].keys())

    if (dataset == "single_sersic") or (dataset == "realistic"):
        bd_together = True
    nb_params = max(2, len(params))
    if (not bd_together) & ("re" in params):
        nb_params += 1
    if (not bd_together) & ("q" in params):
        nb_params += 1
    x_bins = (mag_bins[1:] + mag_bins[:-1]) * 0.5

    nb_cols = 3
    if show_scores:
        nb_cols += 1

    fig, ax = plt.subplots(
        nb_params, nb_cols, figsize=(nb_cols * 5, (nb_params - 0.2) * 5)
    )
    plt.subplots_adjust(wspace=0.35, hspace=0.2)
    [axe.yaxis.set_major_formatter(FormatStrFormatter("%.2f")) for axe in ax.flatten()]
    [axe.grid(True, which="both") for axe in ax[:, :].flatten()]
    [
        axe.hlines(
            0, mag_bins[0] - 0.5, mag_bins[-1] + 0.5, color="red", lw=5, alpha=0.5
        )
        for axe in ax[:, 0].flatten()
    ]
    if x_axis == "True Magnitude":
        [
            axe.set_xlim(mag_bins[0] - 0.5, mag_bins[-1] + 0.5)
            for axe in ax[:, :-1].flatten()
        ]
    else:
        [
            axe.set_xlim(mag_bins[0] - 0.1, mag_bins[-1] + 0.1)
            for axe in ax[:, :-1].flatten()
        ]
    [axe.set_xticklabels([]) for axe in ax[:-1, :-1].flatten()]
    legend_columns = 3

    fs = 20
    [ax[-1, i].set_xlabel(labels[x_axis], fontsize=fs) for i in [0, 1, 2]]
    ax[0, 0].set_title(r"Bias $\mathcal{B}$", fontsize=fs, color="red")
    ax[0, 1].set_title(r"Dispersion $\mathcal{D}$", fontsize=fs, color="red")
    ax[0, 2].set_title("Outlier Fraction $\mathcal{O}$", fontsize=fs, color="red")
    p = 0
    for param in params:
        if show_scores:
            ax[p, -1].set_axis_off()
        if type(global_scores[param][codes[0]]) == list:

            if show_scores:
                ax[p + 1, -1].set_axis_off()

            """ Need to do that more clever"""
            bulge_scores = []
            disk_scores = []

            for code in codes:
                bulge_scores.append(global_scores[param][code][0])
                disk_scores.append(global_scores[param][code][1])
            sorted_scores = [np.sort(bulge_scores)[::-1], np.sort(disk_scores)[::-1]]
        else:
            sorted_scores = np.sort(
                np.fromiter(global_scores[param].values(), dtype=float)
            )[::-1]

        for c, code in enumerate(codes):
            if (
                (dataset == "realistic") | (dataset == "single_sersic") | (param in ["mag", "n", "bt", "BulgeSersic"])
            ):
                ax[p, 0].plot(
                    x_bins,
                    means[param][code],
                    marker=symbols[code],
                    color=colors[code],
                    label=labels[code],
                    markersize=10,
                )
                ax[p, 1].plot(
                    x_bins,
                    stds[param][code],
                    marker=symbols[code],
                    color=colors[code],
                    label=labels[code],
                    markersize=10,
                )
                ax[p, 2].plot(
                    x_bins,
                    outs[param][code],
                    marker=symbols[code],
                    color=colors[code],
                    label=labels[code],
                    markersize=10,
                )
                rank_code = np.where(sorted_scores == global_scores[param][code])[0]
                if show_scores:
                    ax[p, -1].text(
                        0.01,
                        0.1 + rank_code / len(codes),
                        f"$\mathcal{{S}}_{{\mathrm{{{code}}}}}$ = {np.round(global_scores[param][code], 2)}",
                        fontsize=22,
                        color=colors[code],
                    )

            else:
                # bulges
                if param == "re":
                    ax[p, 0].set_yscale("symlog")
                    ax[p, 1].set_yscale("symlog")
                ax[p, 0].plot(
                    x_bins,
                    means[param][code][:][0],
                    marker=symbols[code],
                    color=colors[code],
                    label=labels[code],
                    markersize=10,
                )
                ax[p, 1].plot(
                    x_bins,
                    stds[param][code][:][0],
                    marker=symbols[code],
                    color=colors[code],
                    label=labels[code],
                    markersize=10,
                )
                ax[p, 2].plot(
                    x_bins,
                    outs[param][code][:][0],
                    marker=symbols[code],
                    color=colors[code],
                    label=labels[code],
                    markersize=10,
                )
                rank_code = np.where(sorted_scores[0] == global_scores[param][code][0])[
                    0
                ]
                if show_scores:
                    ax[p, -1].text(
                        0.01,
                        0.1 + rank_code / len(codes),
                        f"$\mathcal{{S}}_{{\mathrm{{{code}}}}}$ = {np.round(global_scores[param][code][0], 2)}",
                        fontsize=22,
                        color=colors[code],
                    )

                if bd_together:
                    disk = mlines.Line2D([], [], color="gray", label="Disks", ls="--")
                    bulges = mlines.Line2D([], [], color="gray", label="Bulges")

                    legend_dc = ax[0, 0].legend(
                        fontsize=fs, handles=[disk, bulges], ncol=2, loc=(0, 1.55)
                    )
                    ax[0, 0].add_artist(legend_dc)

                    # disks
                    ax[p, 0].plot(
                        x_bins,
                        means[param][code][:][1],
                        marker=symbols[code],
                        ls="--",
                        color=colors[code],
                        markersize=10,
                    )
                    ax[p, 1].plot(
                        x_bins,
                        stds[param][code][:][1],
                        marker=symbols[code],
                        ls="--",
                        color=colors[code],
                        markersize=10,
                    )
                    ax[p, 2].plot(
                        x_bins,
                        outs[param][code][:][1],
                        marker=symbols[code],
                        ls="--",
                        color=colors[code],
                        markersize=10,
                    )
                else:  # if separate
                    rank_code = np.where(
                        sorted_scores[1] == global_scores[param][code][1]
                    )[0]
                    if show_scores:
                        ax[p + 1, -1].text(
                            0.01,
                            0.1 + rank_code / len(codes),
                            f"$\mathcal{{S}}_{{\mathrm{{{code}}}}}$ = {np.round(global_scores[param][code][0], 2)}",
                            fontsize=22,
                            color=colors[code],
                        )

                    ax[p + 1, 0].plot(
                        x_bins,
                        means[param][code][:][1],
                        marker=symbols[code],
                        ls="--",
                        color=colors[code],
                        markersize=10,
                    )
                    ax[p + 1, 1].plot(
                        x_bins,
                        stds[param][code][:][1],
                        marker=symbols[code],
                        ls="--",
                        color=colors[code],
                        markersize=10,
                    )
                    ax[p + 1, 2].plot(
                        x_bins,
                        outs[param][code][:][1],
                        marker=symbols[code],
                        ls="--",
                        color=colors[code],
                        markersize=10,
                    )

            ax[0, 0].legend(fontsize=fs, ncol=legend_columns, loc=(0, 1.2))

        if (not bd_together) & (param in ["re", "q"]):
            ax[p, 0].set_ylabel(f"Bulge {labels[param]}", color="red", fontsize=fs)
            ax[p + 1, 0].set_ylabel(f"Disk {labels[param]}", color="red", fontsize=fs)
        else:
            ax[p, 0].set_ylabel(labels[param], color="red", fontsize=fs)
        p += 1
        if (not bd_together) & (param in ["re", "q"]):
            p += 1
    if len(params) == 1:
        ax[-1, 0].set_visible(False)
        ax[-1, 1].set_visible(False)
        ax[-1, 2].set_visible(False)
        [ax[0, i].set_xlabel(labels[x_axis], fontsize=fs) for i in [0, 1, 2]]

    return fig


def summary_plot2D(codes, param, summary2D, mag_bins, bt_bins, labels):

    """
    Create the 2D Summary plot representing the Bias and  Dispersion for a parameter
    regarding the magnitude and the bulge over total flux ratio.

    ## TO DO: remove codes, which could be automatically found in the summary 2D summary.

    Parameters
    ----------

    codes : list of string
        The list of codes names to be plotted.

    param : string
        The parameter name to be plotted.

    summary2D : List of dictionaries, of length 2
        The Bias (summary2D[0]) and dispersion (summary2D[1]) for each code
        Each metric is a dictionary of 2D numpy arrays, with the metrics regarding mag and bt, for each code.

    mag_bins : numpy array
        The bins of magnitude in which the 2D summary has been computed on.

    bt_bins : numpy array
        The bins of bulge over total flux ratio in which the 2D summary has been computed on.

    labels : dictionary
        The corresponding name of the codes, params etc to be plotted instead of the code syntax.
        E.g. labels{"re": "Effective Radius"}

    Returns
    ----------
    fig : The matplotlib Figure object to be plotted in main

    """

    fs_colorbar = 30
    std_cmap = "autumn"
    fig, ax = plt.subplots(len(codes), 2, figsize=(20, 35))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    # minv = min(np.min(summary2D[0]['re'][code][0]), np.min(summary2D[0]['re'][code][1]))
    # maxv = max(np.max(summary2D[0]['re'][code][0]), np.max(summary2D[0]['re'][code][1]))
    fig.suptitle(labels[param], y=0.92, color="red")
    for i, code in enumerate(codes):
        divider = make_axes_locatable(ax[i, 0])
        cax1 = divider.append_axes("right", size="8%", pad=0.8)
        cax2 = divider.append_axes("right", size="8%", pad=0.1)
        nb_bins_bt = np.shape(summary2D[0][param][code])[1]
        nb_bins_mag = np.shape(summary2D[0][param][code])[2]
        x, y = np.meshgrid(np.arange(nb_bins_bt) + 0.5, np.arange(nb_bins_mag) + 0.5)

        if (code in ["gala", "profit"]) & (param == "re"):
            bulges = ax[i, 0].pcolormesh(
                summary2D[0][param][code][0],
                edgecolors="w",
                cmap="Blues",
                norm=LogNorm(),
            )
            std_bulges = ax[i, 0].scatter(
                x,
                y,
                s=100,
                c=summary2D[1][param][code][0],
                cmap=std_cmap,
                norm=LogNorm(),
            )
            min_cbar, max_cbar = np.log10(
                np.min(summary2D[0][param][code][0])
            ), np.log10(np.max(summary2D[0][param][code][0]))
            cbar_ticks = np.logspace(min_cbar, max_cbar, 8)
            min_cbar_std, max_cbar_std = np.log10(
                np.min(summary2D[1][param][code][0])
            ), np.log10(np.max(summary2D[1][param][code][0]))
            cbar_std_ticks = np.logspace(min_cbar_std, max_cbar_std, 8)
        else:
            bulges = ax[i, 0].pcolormesh(
                summary2D[0][param][code][0], edgecolors="w", cmap="Blues"
            )  # np.min(np.max(summary2D[0][param][code][0]), 2))
            std_bulges = ax[i, 0].scatter(
                x, y, s=100, c=summary2D[1][param][code][0], cmap=std_cmap
            )
            min_cbar, max_cbar = np.min(summary2D[0][param][code][0]), np.max(
                summary2D[0][param][code][0]
            )
            min_cbar_std, max_cbar_std = np.min(summary2D[1][param][code][0]), np.max(
                summary2D[1][param][code][0]
            )
            cbar_ticks = np.linspace(min_cbar, max_cbar, 8)
            cbar_std_ticks = np.linspace(min_cbar_std, max_cbar_std, 8)

        ax[i, 0].set_yticks(np.linspace(0, nb_bins_mag, nb_bins_mag + 1))
        ax[i, 0].set_yticklabels(np.round(mag_bins, 1))
        ax[i, 0].set_xticks(np.linspace(0, nb_bins_bt, nb_bins_bt + 1))
        ax[i, 0].set_xticklabels(np.round(bt_bins, 1), rotation=30)
        ax[i, 0].set_title(f"{labels[code]} Bulges", fontsize=20)
        ax[i, 0].set_xlabel("True b/t", fontsize=20)
        ax[i, 0].set_ylabel("True VIS magnitude", fontsize=20)

        cb = plt.colorbar(bulges, ax=ax[i, 0], cax=cax1)
        cb.ax.set_title(r"$\mathcal{B}$", fontsize=fs_colorbar)
        cb.ax.yaxis.set_ticks(cbar_ticks)
        cb.ax.yaxis.set_ticks_position("left")
        cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        cb_std = plt.colorbar(std_bulges, ax=ax[i, 0], cax=cax2)
        cb_std.ax.yaxis.set_ticks(cbar_std_ticks)
        cb_std.ax.set_title(r"$\mathcal{D}$", fontsize=fs_colorbar)
        cb_std.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        """ DISKS """
        divider = make_axes_locatable(ax[i, 1])
        cax1 = divider.append_axes("right", size="8%", pad=0.7)
        cax2 = divider.append_axes("right", size="8%", pad=0.1)

        disks = ax[i, 1].pcolormesh(
            summary2D[0][param][code][1], edgecolors="w", cmap="Blues"
        )  # , vmin=minv, vmax=maxv)
        std_disks = ax[i, 1].scatter(
            x, y, s=100, c=summary2D[1][param][code][1], cmap=std_cmap
        )

        ax[i, 1].set_yticks(np.linspace(0, nb_bins_mag, nb_bins_mag + 1))
        ax[i, 1].set_yticklabels(np.round(mag_bins, 1))
        ax[i, 1].set_xticks(np.linspace(0, nb_bins_bt, nb_bins_bt + 1))
        ax[i, 1].set_xticklabels(np.round(bt_bins, 1), fontsize=15, rotation=30)
        ax[i, 1].set_title(f"{labels[code]} Disks", fontsize=20)
        ax[i, 1].set_xlabel("True b/t", fontsize=20)
        ax[i, 1].set_ylabel("True VIS magnitude", fontsize=20)

        cb2 = plt.colorbar(disks, ax=ax[i, 1], cax=cax1)
        cb2.ax.set_title(r"$\mathcal{B}$", fontsize=fs_colorbar)
        cb2.ax.yaxis.set_ticks(
            np.linspace(
                np.min(summary2D[0][param][code][1]),
                np.max(summary2D[0][param][code][1]),
                8,
            )
        )
        cb2.ax.yaxis.set_ticks_position("left")
        cb2.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        cb2_std = plt.colorbar(std_disks, ax=ax[i, 1], cax=cax2)
        cb2_std.ax.yaxis.set_ticks(
            np.linspace(
                np.min(summary2D[1][param][code][1]),
                np.max(summary2D[1][param][code][1]),
                8,
            )
        )
        cb2_std.ax.set_title(r"$\mathcal{D}$", fontsize=fs_colorbar)
        cb2_std.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    return fig
