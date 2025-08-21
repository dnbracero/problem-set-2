'''
PART 1: ETL
- This code sets up the datasets for Problem Set 2
- NOTE: You will update this code for PART 4: CATEGORICAL PLOTS
'''

# import os 
# change for Path
from pathlib import Path
import pandas as pd
import numpy as np

def create_directories(directories) -> None:
    """
    Creates the necessary directories for storing plots and data.

    Args:
        directories (list of str): A list of directory paths to create.
    """

    # for directory in directories:
    #     os.makedirs(directory, exist_ok=True)
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def extract_transform() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extracts and transforms data from arrest records for analysis

    Returns:
        - `pred_universe`: The dataframe containing prediction-related data for individuals
        - `arrest_events`: The dataframe containing arrest event data
        - `charge_counts`: A dataframe with counts of charges aggregated by charge degree
        - `charge_counts_by_offense`: A dataframe with counts of charges aggregated by both charge degree and offense category
    """
    # Extracts arrest data CSVs into dataframes
    pred_universe = pd.read_csv('https://www.dropbox.com/scl/fi/a2tpqpvkdc8n6advvkpt7/universe_lab9.csv?rlkey=839vsc25njgfftzakr34w2070&dl=1')
    arrest_events = pd.read_csv('https://www.dropbox.com/scl/fi/n47jt4va049gh2o4bysjm/arrest_events_lab9.csv?rlkey=u66usya2xjgf8gk2acq7afk7m&dl=1')

    # Creates two additional dataframes using groupbys
    charge_counts = arrest_events.groupby(['charge_degree']).size().reset_index(name='count')
    charge_counts_by_offense = arrest_events.groupby(['charge_degree', 'offense_category']).size().reset_index(name='count')
    
    return pred_universe, arrest_events, charge_counts, charge_counts_by_offense

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
# Part 4: Categorical Plots
def compute_felony_charge(arrest_events: pd.DataFrame) -> pd.DataFrame:
    """
    Compute whether each arrest had at least one felony charge.

    Parameters
    ----------
    arrest_events : pd.DataFrame
        Requires 'arrest_id' and 'charge_degree'.

    Returns
    -------
    pd.DataFrame
        Columns: ['arrest_id', 'has_felony_charge'] with 0/1 values.
    """
    if 'arrest_id' not in arrest_events.columns:
        raise KeyError("Expected 'arrest_id' column in arrest_events.")
    if 'charge_degree' not in arrest_events.columns:
        raise KeyError("Expected 'charge_degree' column in arrest_events.")

    def is_felony(series: pd.Series) -> int:
        s = series.astype(str).str.lower()
        return int((s.str.contains('felon') | (s == 'f') | (s == 'felony')).any())

    # Returns exactly two columns: ['arrest_id', 'has_felony_charge']
    felony_charge = (
        arrest_events
        .groupby('arrest_id')['charge_degree']
        .apply(is_felony)
        .reset_index(name='has_felony_charge')
    )
    return felony_charge.astype({'has_felony_charge': 'int64'})


def merge_felony_with_pred(pred_universe: pd.DataFrame, felony_charge: pd.DataFrame) -> pd.DataFrame:
    """
    Merge felony-charge indicator with prediction universe.

    Parameters
    ----------
    pred_universe : pd.DataFrame
        Table containing prediction probabilities and identifiers.
    felony_charge : pd.DataFrame
        Output of compute_felony_charge(), keyed by 'arrest_id'.

    Returns
    -------
    pred_universe_w_charge : pd.DataFrame
        pred_universe joined with ['has_felony_charge'] and a friendly 'charge_type' label.
    """
    # Prefer joining on 'arrest_id'
    join_key = 'arrest_id' if 'arrest_id' in pred_universe.columns else 'person_id'
    if join_key not in pred_universe.columns:
        raise KeyError("Expected 'arrest_id' or 'person_id' in pred_universe for the merge.")
    if join_key not in felony_charge.columns:
        # If arrest_id is missing in felony_charge, we cannot merge reliably.
        raise KeyError(f"Expected '{join_key}' column in felony_charge for the merge.")

    df = pred_universe.merge(felony_charge.rename(columns={'arrest_id': join_key}),
                             how='left', on=join_key)
    df['has_felony_charge'] = df['has_felony_charge'].fillna(0).astype(int)
    df['charge_type'] = np.where(df['has_felony_charge'] == 1, 'Felony charge', 'No felony charge')
    return df