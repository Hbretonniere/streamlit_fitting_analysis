import os
import pickle

import numpy as np
import streamlit
from astropy.table import Table


RESULTS_DIR = "results"


def read_catalogue(filename):
    """
    Transform an astropy Table to a panda dat

    Parameters
    ----------
    filename : string
        The name of the catalogue file

    Return
    ----------
    The dataframe containing all the galaxies / codes / fitted parameters

    """
    return Table.read(filename).to_pandas()


@streamlit.cache
def load_data(dataset, band=None, nb_free=False, demo=False):
    """
    Load the wanted catalogues. All the catalogues have the same name structures,
    which can be loaded thanks to the values returned by the different buttons selected in the
    main.

    Parameters
    ----------
    dataset : string
        The name of the dataset you want to study, e.g. 'single_sersic'.

    band : boolean, optional, default False
        If True, load for the wanted band among the multiband catalogues

    nb_free : boolean, optional, default False
        If True, load the double sersic catalogue fitted with a free bulge Sersic index model.
        If False, load the fixed bulge Sersic fit model.

    demo : boolean, default False
        If True, load only 1/100 of the catalogue. Used for exploring the app
        Must be False to have final scientific results

    Return
    ----------
    The dataframe containing all the galaxies / codes / fitted parameters

    """

    nb_free_prefix = "nb_free_" if nb_free else ""
    band = "" if band is None else f"_{band}"
    filename = f"data/{nb_free_prefix}{dataset}{band}.fits"
    cat = read_catalogue(filename)

    if dataset == "realistic":
        cat["Truere"] /= np.sqrt(cat["Trueq"])

    if demo:
        # Use only 10% of the full catalogue
        return cat[::10]
    return cat


def save_results(results, dataset, nb_free):
    """
    Save the metrics results dictionary in a pickle file

    # TO DO: - see what happens for global score plots
             - do the same nb_free but for the band

    Parameters
    ----------
    results : list of dictionary, length 3
        results = [bias, dispersion, outlier fraction]
        each metric is a dictionary, containing the result of the metric for each code and parameters plotted

    dataset : string
        The name of the selected dataset. Used for customizing the name of the saved file.

    nb_free : string
        "True" or "False. Used for customizing the name of the saved file if the free bulge sersic model fit
        option is selected.

    Return
    ----------
    filepath: string
        The name of the file path which will be written below the plot.

    """

    os.makedirs(RESULTS_DIR, exist_ok=True)
    filename = f"results_{dataset}_overlap_True_nbfree_{nb_free}.pickle"
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, "wb") as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return filepath
