'''
- You will run Problem Set 2 from this .py, so make sure to set things up to return outputs accordingly
- Go through each PART and write code / make updates as necessary to produce all required outputs
- Run main.py before you start
'''

import src.part1_etl as part1
import src.part2_plot_examples as part2
import src.part3_bar_hist as part3
import src.part4_catplot as part4
import src.part5_scatter as part5

def main():
    ##  PART 1: ETL  ##
    # ETL the datasets into dataframes
    directories = ['data/part2_plots', 'data/part3_plots', 'data/part4_plots', 'data/part5_plots']
    part1.create_directories(directories)
    
    pred_universe, arrest_events, charge_counts, charge_counts_by_offense = part1.extract_transform()
    print("Part 1 - ETL complete. Files saved:")
    for path in directories:
        print(f" - {path.split('/')[-1]}: {path}")

    # used for parts 4 and parts 5
    felony_charge = part1.compute_felony_charge(arrest_events)
    pred_universe_w_charge = part1.merge_felony_with_pred(pred_universe, felony_charge)
    
    ##  PART 2: PLOT EXAMPLES  ##
    # Apply plot theme
    part2.seaborn_settings()

    # Generate plots
    part2.barplots(charge_counts, charge_counts_by_offense)
    part2.cat_plots(charge_counts, pred_universe)
    part2.histograms(pred_universe)
    part2.scatterplot(pred_universe)
    print('Part 2 - Plot examples complete.')

    ##  PART 3: BAR PLOTS AND HISTOGRAMS  ##
    part3.barplot_fta(pred_universe)
    part3.barplot_fta_by_sex(pred_universe)
    part3.histogram_age(pred_universe)
    part3.histogram_age_binned(pred_universe)
    print('Part 3 - Bar plots and histograms complete.')

    ##  PART 4: CATEGORICAL PLOTS  ##
    part4.catplot_pred_felony_by_charge(pred_universe_w_charge)
    part4.catplot_pred_nonfelony_by_charge(pred_universe_w_charge)
    part4.catplot_pred_felony_by_charge_hue_actual(pred_universe_w_charge)
    print('Part 4 - Categorical plots complete.')

    ##  PART 5: SCATTERPLOTS  ##
    part5.scatter_felony_vs_nonfelony(pred_universe_w_charge)
    part5.scatter_pred_vs_actual_felony(pred_universe_w_charge)
    print('Part 5 - Scatterplots complete.')



if __name__ == "__main__":
    main()
