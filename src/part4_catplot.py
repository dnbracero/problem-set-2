'''
PART 4: CATEGORICAL PLOTS
- Write functions for the tasks below
- Update main() in main.py to generate the plots and print statments when called
- All plots should be output as PNG files to `data/part4_plots`
'''
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

DATA_PLOT_DIR = Path('data/part4_plots')

def save_plot(figpath: Path) -> None:
    # helper function to save plots to data/part3_plots
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, bbox_inches='tight')
    plt.close()

##  UPDATE `part1_etl.py`  ##
# 1. The charge_no column in arrest events tells us the charge degree and offense category for each arrest charge. 
# An arrest can have multiple charges. We want to know if an arrest had at least one felony charge.
# 
# Use groupby and apply with lambda to create a new dataframe called `felony_charge` that has columns: ['arrest_id', 'has_felony_charge']
# 
# Hint 1: One way to do this is that in the lambda function, check to see if a charge_degree is felony, sum these up, and then check if the sum is greater than zero. 
# Hint 2: Another way to do thisis that in the lambda function, use the `any` function when checking to see if any of the charges in the arrest are a felony
# 2. Merge `felony_charge` with `pre_universe` into a new dataframe
# 3. You will need to update ## PART 1: ETL ## in main() to call these two additional dataframes
##  PLOTS  ##
# 1. Create a catplot where the categories are charge type and the y-axis is the prediction for felony rearrest. Set kind='bar'.
def catplot_pred_felony_by_charge(pred_universe_w_charge: pd.DataFrame) -> Path:
    """
    Bar catplot of predicted felony rearrest probability by 'charge_type'.

    Parameters
    ----------
    pred_universe_w_charge : pd.DataFrame
        Dataset containing 'charge_type' and 'prediction_felony' columns.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    for col in ['charge_type', 'prediction_felony']:
        if col not in pred_universe_w_charge.columns:
            raise KeyError(f"Expected '{col}' column in pred_universe_w_charge.")
    sns.catplot(data=pred_universe_w_charge, x='charge_type', y='prediction_felony', kind='bar')
    out = DATA_PLOT_DIR / "catplot_pred_felony_by_charge.png"
    save_plot(out)
    print("Q: How do predicted felony rearrest probabilities differ by charge type?")
    print("A: Arrestees with a current felony charge tend to have higher predicted felony rearrest probabilities than those without a felony charge (as expected).")
    return out

# 2. Now repeat but have the y-axis be prediction for nonfelony rearrest
# 
# In a print statement, answer the following question: What might explain the difference between the plots?
def catplot_pred_nonfelony_by_charge(pred_universe_w_charge: pd.DataFrame) -> Path:
    """
    Bar catplot of predicted non-felony rearrest probability by 'charge_type'.

    Parameters
    ----------
    pred_universe_w_charge : pd.DataFrame
        Dataset containing 'charge_type' and 'prediction_nonfelony' columns.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    for col in ['charge_type', 'prediction_nonfelony']:
        if col not in pred_universe_w_charge.columns:
            raise KeyError(f"Expected '{col}' column in pred_universe_w_charge.")
    sns.catplot(data=pred_universe_w_charge, x='charge_type', y='prediction_nonfelony', kind='bar')
    out = DATA_PLOT_DIR / "catplot_pred_nonfelony_by_charge.png"
    save_plot(out)
    print("Q: What might explain differences between felony vs non-felony prediction plots?")
    print("A: Features correlated with felony charges may be weighted more toward felony risk; non-felony risk can follow different patterns across charge types.")
    return out

# 3. Repeat the plot from 1, but hue by whether the person actually got rearrested for a felony crime
# 
# In a print statement, answer the following question: 
# What does it mean that prediction for arrestees with a current felony charge, 
# but who did not get rearrested for a felony crime have a higher predicted probability than arrestees with a current misdemeanor charge, 
# but who did get rearrested for a felony crime?
def catplot_pred_felony_by_charge_hue_actual(pred_universe_w_charge: pd.DataFrame) -> Path:
    """
    Bar catplot of predicted felony rearrest probability by 'charge_type', hue by actual felony rearrest.

    Notes
    -----
    Attempts to detect a column indicating actual felony rearrest among common names.

    Parameters
    ----------
    pred_universe_w_charge : pd.DataFrame
        Dataset containing 'charge_type', 'prediction_felony', and an outcome column for actual felony rearrest.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    if 'charge_type' not in pred_universe_w_charge.columns or 'prediction_felony' not in pred_universe_w_charge.columns:
        raise KeyError("Expected 'charge_type' and 'prediction_felony' columns in pred_universe_w_charge.")

    possible_actual_cols = ['rearrested_felony', 'y', 'y_felony', 'felony_rearrest', 'actual_felony_rearrest']
    actual_col = next((c for c in possible_actual_cols if c in pred_universe_w_charge.columns), None)
    if actual_col is None:
        raise KeyError(f"Could not find an actual felony rearrest column among: {possible_actual_cols}")

    sns.catplot(
        data=pred_universe_w_charge,
        x='charge_type',
        y='prediction_felony',
        hue=actual_col,
        kind='bar'
    )
    out = DATA_PLOT_DIR / "catplot_pred_felony_by_charge_hue_actual.png"
    save_plot(out)
    print("Q: Interpretation of higher predictions for some who were not rearrested vs lower for some who were?")
    print("A: Predictions are probabilistic. Some high-risk individuals will not reoffend and some low-risk individuals will; the model is not deterministic and errors/noise are expected.")
    return out