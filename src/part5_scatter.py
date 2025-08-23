'''
PART 5: SCATTER PLOTS
- Write functions for the tasks below
- Update main() in main.py to generate the plots and print statments when called
- All plots should be output as PNG files to `data/part5_plots`
'''

from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

DATA_PLOT_DIR = Path('data/part5_plots')

def save_plot(figpath: Path) -> None:
    # helper function to save plots to data/part3_plots
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, bbox_inches='tight')
    plt.close()

# 1. Using lmplot, create a scatter plot where the x-axis is the prediction for felony and the y-axis the is prediction for a nonfelony, and hue this by whether the current charge is a felony. 
# 
# In a print statement, answer the following question: What can you say about the group of dots on the right side of the plot?
def scatter_felony_vs_nonfelony(pred_universe_w_charge: pd.DataFrame) -> Path:
    """
    Scatter (lmplot) of prediction_felony (x) vs prediction_nonfelony (y), hue by 'charge_type'.

    Parameters
    ----------
    pred_universe_w_charge : pd.DataFrame
        Dataset with 'prediction_felony', 'prediction_nonfelony', and 'charge_type' columns.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    need = ['prediction_felony', 'prediction_nonfelony', 'charge_type']
    for col in need:
        if col not in pred_universe_w_charge.columns:
            raise KeyError(f"Expected '{col}' column in pred_universe_w_charge.")
    sp = sns.lmplot(
        data=pred_universe_w_charge,
        x='prediction_felony',
        y='prediction_nonfelony',
        hue='charge_type',
        fit_reg=False
    )
    # Optional diagonal reference
    sp.ax.axline(xy1=(0, 0), xy2=(1, 1), dashes=(2, 2))
    out = DATA_PLOT_DIR / "scatter_felony_vs_nonfelony.png"
    save_plot(out)
    print("Q: What about the dots on the right side of the plot?")
    print("A: Points on the right represent higher predicted felony risk; if many cluster high on x but vary on y, the model differentiates felony vs non-felony risk across individuals.")
    return out

# 2. Create a scatterplot where the x-axis is prediction for felony rearrest and the y-axis is whether someone was actually rearrested.
# 
# In a print statement, answer the following question: Would you say based off of this plot if the model is calibrated or not?
def scatter_pred_vs_actual_felony(pred_universe_w_charge: pd.DataFrame) -> Path:
    """
    Scatter of predicted felony probability vs actual felony rearrest indicator.

    Notes
    -----
    Attempts to detect a column indicating actual felony rearrest among common names.

    Parameters
    ----------
    pred_universe_w_charge : pd.DataFrame
        Dataset with 'prediction_felony' and an outcome column for actual felony rearrest.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    possible_actual_cols = ['rearrested_felony', 'y', 'y_felony', 'felony_rearrest', 'actual_felony_rearrest']
    actual_col = next((c for c in possible_actual_cols if c in pred_universe_w_charge.columns), None)
    if actual_col is None:
        raise KeyError(f"Could not find an actual felony rearrest column among: {possible_actual_cols}")
    sp = sns.lmplot(
        data=pred_universe_w_charge,
        x='prediction_felony',
        y=actual_col,
        fit_reg=False
    )
    out = DATA_PLOT_DIR / "scatter_pred_vs_actual_felony.png"
    save_plot(out)
    print("Q: Based on this plot, is the model calibrated?")
    print("A: Perfect calibration would show average outcomes rising with predicted risk; visually assess whether higher predictions correspond to a higher share of 1s along the y-axis.")
    return out