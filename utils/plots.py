import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors

from inner_functions.names import sources
from inner_types.learning import LearningApproach
from inner_types.plots import FigSize, AxisLabel, FontSize, Legend


def plot_losses(training_losses: np.array, testing_losses: np.array, approach: LearningApproach, path: str):
    plt.figure(figsize=FigSize.WIDER.value)

    plt.ylim([0.0, 0.1])
    plt.yticks(np.arange(0, 0.11, 0.01), fontsize=FontSize.DEFAULT.value)
    plt.ylabel(AxisLabel.LOSS.value, fontsize=FontSize.DEFAULT.value)

    steps = int(len(training_losses) / 10) if len(training_losses) >= 50 else int(len(training_losses) / 5)
    if len(training_losses) <= 10:
        steps = len(training_losses)

    x_values = np.arange(1, len(training_losses) + 1)
    plt.xticks(np.arange(0, len(training_losses) + 1, steps), fontsize=FontSize.DEFAULT.value)

    plt.plot(x_values, training_losses, '-', label='Training losses')
    plt.plot(x_values, testing_losses, '--', label='Testing losses')
    plt.legend(loc=Legend.BEST_LOCATION.value, ncol=Legend.N_COLUMNS_2.value, fontsize=FontSize.DEFAULT.value)

    if approach == LearningApproach.CEN:
        plt.xlabel(AxisLabel.EPOCH.value, fontsize=FontSize.DEFAULT.value)
    else:
        plt.xlabel(AxisLabel.ROUND.value, fontsize=FontSize.DEFAULT.value)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metric(dataframe: pd.DataFrame,
                k_candidates: np.arange,
                ks_chosen: np.array,
                axis_label: AxisLabel,
                file_path: str):
    plt.figure(figsize=FigSize.WIDER.value)
    columns = dataframe.columns  # columns[0]=sources; columns[1]=all_pairs; columns[2:]=k=2|intra_..k=N|inter_community
    index_of_inter = int((len(columns) / 2) + 1)

    intra_columns = columns[2:index_of_inter]
    inter_columns = columns[index_of_inter:]

    # choice indications
    index_aic_choice = np.where(k_candidates == ks_chosen[0])[0][0]
    index_bic_choice = np.where(k_candidates == ks_chosen[1])[0][0]
    index_best_choice = np.where(k_candidates == ks_chosen[2])[0][0]

    # all_pairs curve
    all_pairs = sources()[0]
    lower_bounds = []
    means = []
    upper_bounds = []
    for _ in k_candidates:
        lower_bounds.append(dataframe[all_pairs][0])
        means.append(dataframe[all_pairs][1])
        upper_bounds.append(dataframe[all_pairs][2])

    plt.plot(k_candidates, means, label=all_pairs.capitalize())
    plt.fill_between(k_candidates, lower_bounds, upper_bounds, alpha=0.2)

    # intra_community curve
    intra_community = sources()[1]
    lower_bounds = dataframe.iloc[0][intra_columns].to_numpy().astype(float)
    means = dataframe.iloc[1][intra_columns].to_numpy().astype(float)
    upper_bounds = dataframe.iloc[2][intra_columns].to_numpy().astype(float)

    plt.plot(k_candidates, means, label=intra_community.capitalize())
    plt.fill_between(k_candidates, lower_bounds, upper_bounds, alpha=0.2)

    plt.scatter(k_candidates[index_aic_choice], means[index_aic_choice], color=mcolors.TABLEAU_COLORS.get('tab:orange'))
    plt.text(k_candidates[index_aic_choice], means[index_aic_choice], f'AIC choice (k={ks_chosen[0]})',
             verticalalignment='bottom', horizontalalignment='center')

    plt.scatter(k_candidates[index_bic_choice], means[index_bic_choice], color=mcolors.TABLEAU_COLORS.get('tab:orange'))
    plt.text(k_candidates[index_bic_choice], means[index_bic_choice], f'BIC choice (k={ks_chosen[1]})',
             verticalalignment='top', horizontalalignment='center')

    plt.scatter(k_candidates[index_best_choice], means[index_best_choice], color=mcolors.TABLEAU_COLORS.get('tab:orange'))
    plt.text(k_candidates[index_best_choice], means[index_best_choice], f'Best choice  (k={ks_chosen[2]})',
             verticalalignment='top', horizontalalignment='center')

    # inter_community curve
    inter_community = sources()[2]
    lower_bounds = dataframe.iloc[0][inter_columns].to_numpy().astype(float)
    means = dataframe.iloc[1][inter_columns].to_numpy().astype(float)
    upper_bounds = dataframe.iloc[2][inter_columns].to_numpy().astype(float)

    plt.plot(k_candidates, means, label=inter_community.capitalize())
    plt.fill_between(k_candidates, lower_bounds, upper_bounds, alpha=0.2)

    plt.scatter(k_candidates[index_aic_choice], means[index_aic_choice], color=mcolors.TABLEAU_COLORS.get('tab:green'))
    plt.scatter(k_candidates[index_bic_choice], means[index_bic_choice], color=mcolors.TABLEAU_COLORS.get('tab:green'))
    plt.scatter(k_candidates[index_best_choice], means[index_best_choice], color=mcolors.TABLEAU_COLORS.get('tab:green'))

    plt.ylabel(axis_label.value, fontsize=FontSize.DEFAULT.value)
    plt.xlabel(AxisLabel.K.value, fontsize=FontSize.DEFAULT.value)
    plt.xticks(k_candidates, fontsize=FontSize.DEFAULT.value)
    plt.legend(loc=Legend.BEST_LOCATION.value, ncol=Legend.N_COLUMNS_3.value, fontsize=FontSize.DEFAULT.value)
    plt.tight_layout()

    plt.savefig(file_path)
    plt.close()
