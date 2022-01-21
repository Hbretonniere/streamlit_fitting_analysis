import numpy as np
import streamlit as st


def compute_snr_weights(mag_bins, order=3):
    mags = np.asarray(
        [
            6.000000,
            17.000000,
            18.000000,
            19.000000,
            20.000000,
            21.000000,
            22.000000,
            23.000000,
            24.000000,
            25.000000,
            26.000000,
        ]
    )
    TU_stds = np.array(
        [
            0.000431,
            0.000705,
            0.001537,
            0.002549,
            0.004231,
            0.006543,
            0.011569,
            0.022240,
            0.047734,
            0.110284,
            0.261253,
        ]
    )
    coefs = np.polyfit(mags, TU_stds, order)[::-1]
    weights = 0
    for i, coef in enumerate(coefs):
        weights += coef * mag_bins ** i
    return weights


def compute_score(
    means,
    sigmas,
    outliers,
    completeness,
    weights,
    factors=[2.5, 2.5, 2.5, 1],
    linear_SNR_weight=False,
):
    if linear_SNR_weight:
        weights = np.ones(len(weights))
    k_m, k_s, k_out, k_c = factors
    means = np.abs(means)
    score = k_c * (1 - completeness) + np.sum(
        weights
        * (
            k_m * np.array(means)
            + k_s * (np.array(sigmas))
            + k_out * np.array(outliers)
        )
    )
    return score


def compute_disp(bias, quantile=0.68):
    return np.nanquantile(np.abs(bias) - np.nanmedian(bias), quantile)


def chose_mode(param):
    if param in ["re", "mag", "reb", "red"]:
        mode = "relative"
    # elif param == 'n':
    # mode = 'log'
    else:
        mode = "absolute"

    return mode


def sub_cat(cat, x_axis, x_bins, bin, converge=True):

    indices = np.where((cat[x_axis] > x_bins[bin]) & (cat[x_axis] < x_bins[bin + 1]))[0]
    nb_gal = len(cat)
    fraction = len(indices) / nb_gal

    return cat.iloc[indices], fraction


# @st.cache
def sub_cat2d(cat, x_axis, x_bins, y_axis, y_bins, xbin, ybin):
    indices = np.where(
        (cat[x_axis] > x_bins[xbin])
        & (cat[x_axis] < x_bins[xbin + 1])
        & (cat[y_axis] > y_bins[ybin])
        & (cat[y_axis] < y_bins[ybin + 1])
    )[0]

    nb_gal = 275664
    fraction = len(indices) / nb_gal

    return cat.iloc[indices], fraction


def compute_error_bin(cat, code, param, mode="absolute", abs=True):

    true = f"True{param}"
    pred = f"Pred{param}_{code}"
    param_pred = cat[pred]
    param_true = cat[true]

    if (code == "software3") & (param == "BulgeSersic"):
        print("change log n sersic bulge software3")
        param_pred = np.power(10, param_pred)

    error = param_pred - param_true
    if abs:
        error = np.abs(error)
    if mode == "log":
        error = (np.log10(cat[pred]) - np.log10(cat[true])) / np.log10(cat[true])
    if mode == "relative":
        error /= param_true
    return error


def find_outliers(cat, code, param, mode, outlier_limit, abs=True):

    error = cat[f"Pred{param}_{code}"] - cat[f"True{param}"]

    if abs:
        error = np.abs(error)

    if mode == "relative":
        relative_error = error / cat[f"True{param}"]
        indices_out = np.where(np.abs(relative_error) > outlier_limit)[0]

    elif mode == "absolute":
        indices_out = np.where(np.abs(error) > outlier_limit)[0]

    return indices_out


@st.cache
def compute_summary(
    cat,
    params,
    codes,
    x_bins,
    outlier_limit=None,
    dataset="single_sersic",
    factors=[1, 1, 1],
    x_axis="Truemag",
    abs=True,
    linear_SNR_weight=False,
):
    """
    Compute summary statistics of different parameters, for given software.

    Parameters
    ----------
    cat :
        The catalogue containing all the software2xies of all the codes
    params : list of string.
        The names of the parameters you want to study.
        Should match the names of the homogenized catalogues.
    codes : list of strings
        The different codes you want to study.
        Should be one of the following : sex+, software3, software2, software4, metryka.
    x_bins : list of float
        The list of bins of the x-axis (magnitude or b/t) for the study. Stop at x_bins[-1].
    outlier_limit : float. The value of the outlier threshold. Objects which have an error bigger than this number
                        will be rejected during the study. If None, no outliers are removed.
    dataset : string. The type of fit you want to study. Must be "single_sersic", "double_sersic", or "realistic".

    Returns
    -------
    code_means : dict
        The different means errors. There is a key by parameter name,
        and a key by code. in each, the list of mean error by bins of magnitude.
        For example, code_means['software2']['re'] will give you the list of mean errors by bin of magnitude
        made by software2pagos on the fitting of the radius.
    code_stds : dict
        Same but for stds.
    code_outs : dict
        Same but for the fraction of outliers.

    Example
    -------
    >>>  compute_summary(cat, ['mag', 're', 'q'], ['software2', 'software3'], np.linspace(18, 25, 2), outlier_limit=0.5, dataset='single_sersic')
    """

    # Initialize the outputs
    params_means, params_stds, params_outs, params_scores, params_scores_global = (
        {},
        {},
        {},
        {},
        {},
    )
    completenesses = {
        "single_sersic": {
            "software1": 0.95,
            "software2": 0.9,
            "software3": 0.98,
            "software4": 0.91,
            "software5": 0.89,
        },
        "double_sersic": {"software1": 0.95, "software2": 0.93, "software3": 0.98, "software4": 0.95},
        "multiband": {"software1": 0.95, "software2": 0.93, "software3": 0.98, "software4": 0.95},
        "realistic": {"software1": 0.85, "software2": 0.71, "software3": 0.92},
    }

    #  Loop through the parameters you want to study
    for param in params:

        # Create the dictionnary corresponding to the parameter
        (
            params_means[param],
            params_stds[param],
            params_outs[param],
            params_scores[param],
            params_scores_global[param],
        ) = ({}, {}, {}, {}, {})

        mode = chose_mode(param)
        # Loop through the codes you want to study

        for c, code in enumerate(codes):
            # Create the list for the mean and stds of the particular code and param (always double component and then remove the disks if its ss)
            means = np.zeros((2, len(x_bins) - 1))
            stds = np.zeros_like(means)
            outs = np.zeros_like(means)
            scores = np.zeros_like(means)
            fracgals = []
            gal_bin = []
            # Loop through the magnitude bins'''
            for i in range(len(x_bins) - 1):
                cat_bin, fraction = sub_cat(cat, x_axis, x_bins, i)
                fracgals.append(fraction)
                gal_bin.append(len(cat_bin))
                if (
                    (dataset == "realistic")
                    | (dataset == "single_sersic")
                    | (param == "bt")
                    | (param == "BulgeSersic")
                    | (param == "mag")
                ):

                    error = compute_error_bin(cat_bin, code, param, mode=mode, abs=abs)
                    if len(error) == 0:
                        raise RuntimeError(
                            f"No software2xy in the bin of magnitude {x_bins[i]-x_bins[i+1]:.2f}, try reducing the magnitude range."
                        )  # assert len(error) != 0, f'No software2xy in the bin of magnitude {x_bins[i]-x_bins[i+1]:.2f}, try reducing the magnitude range.'
                    outliers = find_outliers(
                        cat_bin, code, param, mode, outlier_limit, abs=abs
                    )
                    means[0, i] = np.nanmedian(error)
                    stds[0, i] = compute_disp(error)
                    outs[0, i] = (
                        len(cat_bin) - len(error.drop(error.index[outliers]))
                    ) / len(cat_bin)
                    scores[0, i] = (
                        factors[0] * means[0, i]
                        + factors[1] * stds[0, i]
                        + factors[2] * outs[0, i]
                    )

                else:
                    # indices_bt = np.where(cat_bin['Truebt'] < 0.2)[0]
                    # sub_cat_bulges = cat_bin.drop(cat_bin.index[indices_bt])
                    error_b = compute_error_bin(
                        cat_bin, code, param + "b", mode=mode, abs=abs
                    )
                    outliers_b = find_outliers(
                        cat_bin, code, param + "b", mode, outlier_limit, abs=abs
                    )

                    error_d = compute_error_bin(
                        cat_bin, code, param + "d", mode=mode, abs=abs
                    )
                    outliers_d = find_outliers(
                        cat_bin, code, param + "d", mode, outlier_limit, abs=abs
                    )

                    means[0, i] = np.nanmedian(error_b)
                    means[1, i] = np.nanmedian(error_d)

                    stds[0, i] = compute_disp(error_b)
                    stds[1, i] = compute_disp(error_d)

                    scores[0, i] = (
                        factors[0] * means[0, i]
                        + factors[1] * stds[0, i]
                        + factors[2] * outs[0, i]
                    )
                    scores[1, i] = (
                        factors[0] * means[1, i]
                        + factors[1] * stds[1, i]
                        + factors[2] * outs[1, i]
                    )

                    outs[0, i] = (
                        len(cat_bin) - len(error_b.drop(error_b.index[outliers_b]))
                    ) / len(cat_bin)
                    outs[1, i] = (
                        len(cat_bin) - len(error_d.drop(error_d.index[outliers_d]))
                    ) / len(cat_bin)

            snr_weights = compute_snr_weights(x_bins[:-1])
            weights = gal_bin[: len(x_bins) - 1] * snr_weights
            weights /= np.sum(weights)
            weights = weights[: len(x_bins)]
            if (
                (dataset == "single_sersic")
                | (dataset == "realistic")
                | (param == "bt")
                | (param == "mag")
                | (param == "BulgeSersic")
            ):
                means = means[0]
                stds = stds[0]
                outs = outs[0]
                scores = scores[0]
                completeness = completenesses[dataset][code]
                params_scores_global[param][code] = compute_score(
                    means,
                    stds,
                    outs,
                    completeness,
                    weights,
                    factors=factors,
                    linear_SNR_weight=linear_SNR_weight,
                )

            else:
                params_scores_global[param][code] = []
                completeness = completenesses[dataset][code]
                params_scores_global[param][code].append(
                    compute_score(
                        means[0],
                        stds[0],
                        outs[0],
                        completeness,
                        weights,
                        factors=factors,
                        linear_SNR_weight=linear_SNR_weight,
                    )
                )
                params_scores_global[param][code].append(
                    compute_score(
                        means[1],
                        stds[1],
                        outs[1],
                        completeness,
                        weights,
                        factors=factors,
                        linear_SNR_weight=linear_SNR_weight,
                    )
                )
            # At the end of the loop in the mag bins, add the list to the good code and param key of the dictionnary
            params_means[param][code] = means
            params_stds[param][code] = stds
            params_outs[param][code] = outs
            params_scores[param][code] = scores

    return [params_means, params_stds, params_outs, params_scores, params_scores_global]


def compute_summary2D(
    cat,
    params,
    codes,
    x_bins,
    y_bins,
    outlier_limit=None,
    x_axis="Truemag",
    y_axis="Truebt",
    factors=[20, 0.6, 5],
    abs=True,
):
    """
    Compute the Means, the Standard Deviation and the fraction of outliers of errors in the fitting of different parameters, for different codes.

    Inputs:
            - cat : the catalogue containing all the software2xies of all the codes

            - params : list of strings. The names of the parameters you want to study.
                                        Should match the names of the homogenized catalogues.

            - codes : list of strings. The different codes you want to study.
                                        should be one of the following : sex+, software3, software2, software4, metryka.

            - x_bins : list of float. The list of bins of the x-axis (magnitude or b/t) for the study. Stop at x_bins[-1].

            - outlier_limit : float. The value of the outlier threshold. Objects which have an error bigger than this number
                               will be rejected during the study. If None, no outliers are removed.

            - dataset : string. The type of fit you want to study. Must be "single_sersic", "double_sersic", or "realistic".

    Outputs :
            - code_means : dictionnary. The different means errors. There is a key by parameter name,
                                        and a key by code. in each, the list of mean error by bins of magntiude.
                                        For example, code_means['software2']['re'] will give you the list of mean errors by bin of magnitude
                                        made by software2pagos on the fitting of the radius.

            - code_stds : dictionnary. Same but for stds.

            - code_outs : dictionnary. Same but for the fraction of outliers.

    Example of use :

          compute_summary(cat, ['mag', 're', 'q'], ['software2', 'software3'], np.linspace(18, 25, 2), outlier_limit=0.5, dataset='single_sersic')
    """

    # Initialize the outputs
    params_means, params_stds, params_outs, params_scores = {}, {}, {}, {}
    write = 0
    for param in params:
        (
            params_means[param],
            params_stds[param],
            params_outs[param],
            params_scores[param],
        ) = ({}, {}, {}, {})
        mode = chose_mode(param)

        # Loop through the codes you want to study
        for c, code in enumerate(codes):

            # Create the list for the mean and stds of the particular code and param
            means = np.zeros((2, len(x_bins) - 1, len(y_bins) - 1))
            stds = np.zeros_like(means)
            outs = np.zeros_like(means)
            fracgals = []

            # Loop through the magnitude bins'''
            for i in range(len(x_bins) - 1):
                for j in range(len(y_bins) - 1):
                    try:
                        cat_bin, fraction = sub_cat2d(
                            cat, x_axis, x_bins, y_axis, y_bins, i, j
                        )
                        fracgals.append(fraction)

                        error_b = compute_error_bin(
                            cat_bin, code, param + "b", mode=mode, abs=abs
                        )
                        # outliers_b = find_outliers(cat_bin, code, param+'b', mode, outlier_limit, abs=abs)
                        # error_b.drop(error_b.index[outliers_b], inplace=True)

                        error_d = compute_error_bin(
                            cat_bin, code, param + "d", mode=mode, abs=abs
                        )
                        # outliers_d = find_outliers(cat_bin, code, param+'d', mode, outlier_limit, abs=abs)
                        # error_d.drop(error_d.index[outliers_d], inplace=True)

                        means[0, i, j] = np.nanmedian(error_b)
                        means[1, i, j] = np.nanmedian(error_d)

                        stds[0, i, j] = compute_disp(error_b)
                        stds[1, i, j] = compute_disp(error_d)

                        outb = (len(cat_bin) - len(error_b)) / len(cat_bin)
                        outd = (len(cat_bin) - len(error_d)) / len(cat_bin)
                        outs[0, i, j] = outb
                        outs[1, i, j] = outd
                    except ZeroDivisionError:
                        if write == 0:
                            st.markdown(
                                "## Not enough point on the bins, try a higher minimum magnitude."
                            )
                        write = 1

                params_means[param][code] = means
                params_stds[param][code] = stds
                params_outs[param][code] = outs

    return [params_means, params_stds, params_outs]


def format_func(value):
    return np.round(value, 2)


def compute_error_prediction(cat, params, codes, x_bins, nb_bins):
    calib_mag = {}
    for code in codes:
        calib_mag[code] = {}
        for param in params:
            ins = []
            calibs = []
            calib_mag[param] = np.zeros(nb_bins)
            for i in range(len(x_bins) - 1):
                bin_cat, _ = sub_cat(cat, "Truemag", x_bins, i)
                in_ = []
                for i, err in enumerate(bin_cat[f"Pred{param}err_{code}"]):
                    predm = bin_cat[f"Pred{param}_{code}"].iloc[i]
                    truem = bin_cat[f"True{param}"].iloc[i]
                    if (truem <= predm + err) & (truem > predm - err):
                        in_.append(i)
                ins.append(len(in_))
                fraction = len(in_) / len(bin_cat[f"Pred{param}err_{code}"])
                calibs.append(fraction)

            calibs.append(sum(ins) / len(cat))
            calib_mag[code][param] = calibs

    return calib_mag
