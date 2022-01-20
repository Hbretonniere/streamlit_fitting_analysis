single_sersic_params = {
    "available_params": {
        "Summary Plots": ["re", "q", "n"],
        "Summary Scores": ["re", "q", "n"],
        "Trumpet Plots": ["re", "q", "n"],
        "Error Prediction": ["re", "q", "n"],
    },
    "available_codes": ["deepleg", "gala", "metryka", "profit", "SE++"],
    "plot_types": [
        "Summary Plots",
        "Trumpet Plots",
        "Summary Scores",
        "Error Prediction",
    ],
    "default bins": {
        "Summary Plots": 6,
        "2D Summary Plots": 5,
        "Summary Scores": 6,
        "Trumpet Plots": 6,
        "Error Prediction": 4,
    },
}

double_sersic_params = {
    "available_params": {
        "Summary Plots": ["re", "q", "bt"],
        "2D Summary Plots": ["re", "q"],
        "Summary Scores": ["re", "q", "bt"],
        "Trumpet Plots": ["reb", "red", "qb", "qd", "bt"],
        "Error Prediction": ["reb", "red", "qb", "qd"],
    },
    "available_codes": ["deepleg", "gala", "profit", "SE++"],
    "plot_types": [
        "Summary Plots",
        "2D Summary Plots",
        "Trumpet Plots",
        "Summary Scores",
        "Error Prediction",
    ],
    "default bins": {
        "Summary Plots": 6,
        "2D Summary Plots": 5,
        "Summary Scores": 6,
        "Trumpet Plots": 6,
        "Error Prediction": 4,
    },
}

multiband_params = {
    "available_params": {
        "Summary Plots": ["re", "q", "bt"],
        "2D Summary Plots": ["re", "q"],
        "Summary Scores": ["re", "q", "bt"],
        "Trumpet Plots": ["reb", "red", "qb", "qd", "bt"],
        "Error Prediction": ["reb", "red", "qb", "qd"],
    },
    "available_codes": ["gala", "profit", "SE++"],
    "plot_types": [
        "Summary Plots",
        "2D Summary Plots",
        "Trumpet Plots",
        "Summary Scores",
        "Error Prediction",
    ],
    "default bins": {
        "Summary Plots": 6,
        "2D Summary Plots": 5,
        "Summary Scores": 6,
        "Trumpet Plots": 6,
        "Error Prediction": 4,
    },
}

realistic_params = {
    "available_params": {
        "Summary Plots": ["re", "q", "n"],
        "Summary Scores": ["re", "q", "n"],
        "Trumpet Plots": ["re", "q", "n"],
        "Error Prediction": ["re", "q", "n"],
    },
    "available_codes": ["gala", "profit", "SE++"],
    "plot_types": [
        "Summary Plots",
        "Trumpet Plots",
        "Summary Scores",
        "Error Prediction",
    ],
    "default bins": {
        "Summary Plots": 6,
        "2D Summary Plots": 5,
        "Summary Scores": 6,
        "Trumpet Plots": 6,
        "Error Prediction": 4,
    },
}


double_sersic_free_params = {
    "available_params": {
        "Summary Plots": ["re", "q", "BulgeSersic", "bt"],
        "2D Summary Plots": ["re", "q"],
        "Summary Scores": ["re", "q", "bt"],
        "Trumpet Plots": ["reb", "red", "qb", "qd", "bt"],
        "Error Prediction": ["reb", "red", "qb", "qd"],
    },
    "available_codes": ["gala", "profit", "SE++"],
    "plot_types": [
        "Summary Plots",
        "2D Summary Plots",
        "Trumpet Plots",
        "Summary Scores",
        "Error Prediction",
    ],
    "default bins": {
        "Summary Plots": 6,
        "2D Summary Plots": 5,
        "Summary Scores": 6,
        "Trumpet Plots": 6,
        "Error Prediction": 4,
    },
}

LABELS = {
    "single_sersic": "Single Sersic",
    "double_sersic": "Double Sersic",
    "realistic": "Realistic",
    "multiband": "Multi-band",
    "mag": "Magnitude",
    "re": "Effective radius",
    "reb": "Bulge radius",
    "red": "Disk radius",
    "q": "Axis ratio",
    "qb": "Bulge axis ratio",
    "qd": "Disk axis ratio",
    "n": "Sersic index",
    "BulgeSersic": "Bulge Sersic index",
    "bt": "Bulge \n flux ratio",
    "SE++": "SourceXtractor++",
    "gala": "Galapagos-2",
    "deepleg": "DeepLeGATo",
    "profit": "ProFit",
    "metryka": "Morfometryka",
    "True Magnitude": "VIS True magnitude",
    "Truemag": "VIS True magnitude",
    "True b/t": "VIS True b/t",
    "mu": "mu",
    "Bulge re": "Bulge \n radius",
    "Disk re": "Disk \n radius",
    "Bulge q": "Bulge \n axis ratio",
    "Disk q": "Disk \n axis ratio",
    "Bulge mu": r"Bulge $\mu$",
    "Disk mu": r"Disk $\mu$",
}
