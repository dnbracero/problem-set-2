'''
PART 3: BAR PLOTS AND HISTOGRAMS
- Write functions for the tasks below
- Update main() in main.py to generate the plots and print statments when called
- All plots should be output as PNG files to `data/part3_plots`
'''
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

DATA_PLOT_DIR = Path('data/part3_plots')

def save_plot(figpath: Path) -> None:
    # Helper function to save plots to data/part3_plots
    figpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(figpath, bbox_inches='tight')
    plt.close()

# 1. Using the pre_universe data frame, create a bar plot for the fta column.
def barplot_fta(pred_universe: pd.DataFrame) -> Path:
    """
    Create a bar plot for the 'fta' column (Failure To Appear) and save to PNG.

    Parameters
    ----------
    pred_universe : pd.DataFrame
        Dataset containing an 'fta' column.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    if 'fta' not in pred_universe.columns:
        raise KeyError("Expected 'fta' column in pred_universe.")
    sns.countplot(data=pred_universe, x='fta')
    out = DATA_PLOT_DIR / "fta_barplot.png"
    save_plot(out)
    return out


# 2. Hue the previous barplot by sex
def barplot_fta_by_sex(pred_universe: pd.DataFrame) -> Path:
    """
    Create a bar plot for 'fta' with hue by 'sex' and save to PNG.

    Parameters
    ----------
    pred_universe : pd.DataFrame
        Dataset containing 'fta' and 'sex' columns.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    for col in ['fta', 'sex']:
        if col not in pred_universe.columns:
            raise KeyError(f"Expected '{col}' column in pred_universe.")
    sns.countplot(data=pred_universe, x='fta', hue='sex')
    out = DATA_PLOT_DIR / "fta_barplot_by_sex.png"
    save_plot(out)
    return out


# 3. Plot a histogram of age_at_arrest
def histogram_age(pred_universe: pd.DataFrame) -> Path:
    """
    Plot histogram of 'age_at_arrest' and save to PNG.

    Parameters
    ----------
    pred_universe : pd.DataFrame
        Dataset containing 'age_at_arrest' column.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    if 'age_at_arrest' not in pred_universe.columns:
        raise KeyError("Expected 'age_at_arrest' column in pred_universe.")
    sns.histplot(data=pred_universe, x='age_at_arrest')
    out = DATA_PLOT_DIR / "age_histogram.png"
    save_plot(out)
    return out


# 4. Plot the same histogram, but create bins that represent the following age groups 
def histogram_age_binned(pred_universe: pd.DataFrame) -> Path:
    """
    Plot histogram of 'age_at_arrest' using custom bins and save to PNG.

    Bins: (age groups)
    - 18 to 21
    - 21 to 30
    - 30 to 40
    - 40 to 100

    Parameters
    ----------
    pred_universe : pd.DataFrame
        Dataset containing 'age_at_arrest' column.

    Returns
    -------
    Path
        Path to the saved PNG file.
    """
    if 'age_at_arrest' not in pred_universe.columns:
        raise KeyError("Expected 'age_at_arrest' column in pred_universe.")
    sns.histplot(data=pred_universe, x='age_at_arrest', bins=[18, 21, 30, 40, 100])
    out = DATA_PLOT_DIR / "age_histogram_binned.png"
    save_plot(out)
    return out