import streamlit as st

from app.io import load_data, save_results
from app.help import readme
from app.params import (
    single_sersic_params,
    double_sersic_params,
    double_sersic_free_params,
    realistic_params,
    multiband_params,
    LABELS,
)
from app.summary import summary, summary2D, trumpet, score, error_calibration

DATASETS = ("single_sersic", "double_sersic", "realistic", "multiband")
PARAMETERS_2D = ["re", "q"]


def main():
    """
    Create the different buttons and setting of the app. The description of the buttons are in help.py
    The buttons and setting depend of the selected dataset. The differences are written in the dictionaries
    defined in params.py

    Load the wanted dataset, defined in io.py

    Call the different actions defined in summary.py,
    which call someroutines of utils.py and the plotting routines defined in plot.py

    Launch the web page with the interface
    """
    nb_free = False
    band = None

    description = st.expander("README")
    description.markdown(readme)

    st.title("MorphoChallenge DIY plots")
    demo = st.checkbox(
        "Demo version (much faster). Uncheck when all set to get the full results.",
        value=True,
    )

    st.sidebar.markdown("## Controls")
    st.sidebar.markdown(
        "Adjust the values below and the figures will be updated accordingly"
    )

    dataset = st.sidebar.radio("Select a Dataset", DATASETS, format_func=lambda x: LABELS[x])
    if dataset == "single_sersic":
        dataset_params = single_sersic_params
    elif dataset == "realistic":
        dataset_params = realistic_params
    elif dataset == "double_sersic":
        dataset_params = double_sersic_params
        nb_free = st.sidebar.checkbox("Use free bulge Sersic fit")
    elif dataset == "multiband":
        band = st.sidebar.radio("Which fitted band ?", ["VIS", "NIR-y"])
        dataset_params = multiband_params
        if band == "NIR-y" and "SE++" in dataset_params["available_codes"]:
            dataset_params["available_codes"].remove("SE++")

    if nb_free:
        dataset_params = double_sersic_free_params

    df = load_data(dataset, nb_free=nb_free, band=band, demo=demo)

    plot_type = st.sidebar.radio("Select a Type of plot", dataset_params["plot_types"])

    #  ### PARAMETERS OPTIONS ####
    all_params = st.sidebar.checkbox("Plot all parameters")
    if all_params:
        params = dataset_params["available_params"][plot_type]
    else:
        params = st.sidebar.multiselect(
            "Select relevant parameters",
            dataset_params["available_params"][plot_type],
            default=[
                dataset_params["available_params"][plot_type][0],
                dataset_params["available_params"][plot_type][1],
            ],
            format_func=lambda x: LABELS[x],
        )
        if len(params) == 0:
            st.markdown("## Choose at least one parameter to plot !")
            return 0

    # #####  SOFTWARE OPTIONS ####
    all_code = st.sidebar.checkbox("Plot all software")
    if all_code:
        codes = dataset_params["available_codes"]
    else:
        codes = st.sidebar.multiselect(
            "Select software to display",
            dataset_params["available_codes"],
            default=dataset_params["available_codes"],
            format_func=lambda x: LABELS[x],
        )
        if len(codes) == 0:
            st.markdown("## Select at least one software to plot !")
            return 0

    # #### OUTLIERS OPTIONS ####
    outliers = st.slider(
        "Outliers Threshold", min_value=0.0, max_value=2.0, value=0.5, step=0.05
    )

    #  #### X AXIS OPTIONS ####

    if plot_type == "2D Summary Plots":
        default_bins = 5
        x_axis = "Truemag"
        mag_min, mag_max = st.slider(
            "VIS True magnitude range",
            min_value=18.0,
            max_value=26.0,
            value=[18.5, 25.3],
            step=0.05,
        )
        mags = [mag_min, mag_max]
        bt_min, bt_max = st.slider(
            "bt range", min_value=0.0, max_value=1.0, value=[0.1, 0.9], step=0.1
        )
        bts = [bt_min, bt_max]

        n_bins_bt = st.sidebar.number_input(
            "Number of bt bins", min_value=5, max_value=10, value=default_bins
        )
    else:
        default_bins = 6
        if dataset == "double_sersic":
            x_axis = st.sidebar.radio("X axis", ("Truemag", "Truebt"))
        else:
            x_axis = "Truemag"

        if plot_type == "Trumpet Plots":
            y_max = st.slider(
                "Bias range", min_value=0.3, max_value=5.0, value=1.0, step=0.1
            )

        if x_axis == "Truemag":
            if dataset == "realistic":
                min_mag = 21.0 if demo else 20.0
            elif dataset != "realistic":
                min_mag = 19.0 if demo else 18.5
            x_button_name = "VIS True magnitude range"
            min_val, max_val, step = 18.0, 26.0, 0.5
            values = (min_mag, 25.3)
        elif x_axis == "Truebt":
            x_button_name = "True b/t range"
            min_val, max_val, step = 0.0, 1.0, 0.1
            values = (0.0, 1.0)

        x_min, x_max = st.slider(
            x_button_name, min_value=min_val, max_value=max_val, value=values, step=step
        )
        xs = [x_min, x_max]
    n_bins = st.sidebar.number_input(
        f"Number of {x_axis[4:]} bins",
        min_value=2,
        max_value=15,
        value=dataset_params["default bins"][plot_type],
    )

    # #### More options ####
    more_options = st.sidebar.checkbox("More options")
    absolute = False
    factors = [1, 1, 1, 1]
    show_scores = False
    bd_together = False
    if more_options:
        cut_outliers = st.sidebar.checkbox("Do not remove outliers")
        if (plot_type == "Summary Plots"):
            show_scores = st.sidebar.checkbox("Show_scores")
            if(dataset == "double_sersic"):
                bd_together = st.sidebar.checkbox("Plot bulge and disks together")
        if cut_outliers:
            outliers = 1000
        absolute = st.sidebar.checkbox("Absolute error")

    # #### CREATE AND PLOT ####
    if plot_type == "Summary Plots":
        results = summary(
            df,
            dataset,
            params,
            codes,
            xs,
            n_bins,
            outliers,
            x_axis,
            score_factors=factors,
            abs=absolute,
            show_scores=show_scores,
            bd_together=bd_together,
        )
    elif plot_type == "Trumpet Plots":
        results = trumpet(
            df, params, codes, xs, n_bins, outliers, y_max
        )
    elif plot_type == "2D Summary Plots":
        results = summary2D(
            df,
            params,
            codes,
            mags,
            n_bins,
            bts,
            n_bins_bt,
            outliers
        )
    elif plot_type == "Summary Scores":
        results = score(
            df,
            dataset,
            params,
            codes,
            factors,
            xs,
            n_bins,
            outliers,
            x_axis,
            abs=False,
            show_scores=show_scores
        )
    elif plot_type == "Error Prediction":
        results = error_calibration(df, dataset, params, codes, xs, n_bins)

    if st.button("Save results", disabled="Summary" not in plot_type):
        filepath = save_results(results, dataset, nb_free)
        st.success(f"Results saved as {filepath}")


if __name__ == "__main__":
    main()
