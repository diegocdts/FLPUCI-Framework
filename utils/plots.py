import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from scipy import interpolate

from inner_functions.names import sources, choice_method
from inner_types.learning import LearningApproach
from inner_types.plots import FigSize, AxisLabel, FontSize, Legend


def colors(color_name: str):
    return mcolors.TABLEAU_COLORS.get(f'tab:{color_name}')


def y_tick_2decimal():
    locs, labels = plt.yticks()
    y_ticks = np.array([])
    for idx in np.linspace(locs[0], locs[-1], num=5):
        y_ticks = np.append(y_ticks, '{:.2f}'.format(idx))
    plt.yticks(y_ticks.astype(float), fontsize=FontSize.DEFAULT.value)


def bspline_plot(x: np.array, y: np.array):
    x_values = np.linspace(int(x[0]), int(x[-1]), 100)
    bspline = interpolate.make_interp_spline(x, y, 2)
    y_values = bspline(x_values)
    return x_values, y_values


def fill_between(x_values, lower_bounds, means, upper_bounds, label, kwargs=None):
    x, lower_bounds = bspline_plot(x_values, lower_bounds)
    x, means = bspline_plot(x_values, means)
    x, upper_bounds = bspline_plot(x_values, upper_bounds)

    kwargs = {} if kwargs is None else kwargs

    plt.plot(x, means, label=label.replace('_', ' '), **kwargs)
    plt.fill_between(x, lower_bounds, upper_bounds, alpha=0.2)


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

    y_tick_2decimal()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_metric(dataframe: pd.DataFrame,
                k_candidates: np.arange,
                ks_chosen: np.array,
                axis_label: AxisLabel,
                path: str):
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

    fill_between(k_candidates, lower_bounds, means, upper_bounds, all_pairs.capitalize())

    # intra_community curve
    intra_community = sources()[1]
    lower_bounds = dataframe.iloc[0][intra_columns].to_numpy().astype(float)
    means = dataframe.iloc[1][intra_columns].to_numpy().astype(float)
    upper_bounds = dataframe.iloc[2][intra_columns].to_numpy().astype(float)

    fill_between(k_candidates, lower_bounds, means, upper_bounds, intra_community.capitalize())

    plt.scatter(k_candidates[index_aic_choice], means[index_aic_choice], color=colors('orange'))
    plt.text(k_candidates[index_aic_choice], means[index_aic_choice], f'AIC choice (k={ks_chosen[0]})',
             verticalalignment='bottom', horizontalalignment='left')

    plt.scatter(k_candidates[index_bic_choice], means[index_bic_choice], color=colors('orange'))
    plt.text(k_candidates[index_bic_choice], means[index_bic_choice], f'BIC choice (k={ks_chosen[1]})',
             verticalalignment='top', horizontalalignment='left')

    plt.scatter(k_candidates[index_best_choice], means[index_best_choice], color=colors('orange'))
    plt.text(k_candidates[index_best_choice], means[index_best_choice], f'Best choice  (k={ks_chosen[2]})',
             verticalalignment='center', horizontalalignment='right')

    # inter_community curve
    inter_community = sources()[2]
    lower_bounds = dataframe.iloc[0][inter_columns].to_numpy().astype(float)
    means = dataframe.iloc[1][inter_columns].to_numpy().astype(float)
    upper_bounds = dataframe.iloc[2][inter_columns].to_numpy().astype(float)

    fill_between(k_candidates, lower_bounds, means, upper_bounds, inter_community.capitalize())

    plt.scatter(k_candidates[index_aic_choice], means[index_aic_choice], color=colors('green'))
    plt.scatter(k_candidates[index_bic_choice], means[index_bic_choice], color=colors('green'))
    plt.scatter(k_candidates[index_best_choice], means[index_best_choice], color=colors('green'))

    plt.ylabel(axis_label.value, fontsize=FontSize.DEFAULT.value)
    plt.xlabel(AxisLabel.K.value, fontsize=FontSize.DEFAULT.value)
    plt.xticks(k_candidates, fontsize=FontSize.DEFAULT.value)
    plt.legend(loc=Legend.BEST_LOCATION.value, ncol=Legend.N_COLUMNS_3.value, fontsize=FontSize.DEFAULT.value)

    y_tick_2decimal()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_strategy_comparison(all_pairs_mean: float,
                             acc_means: pd.DataFrame,
                             sli_means: pd.DataFrame,
                             acc_ks_chosen: np.array,
                             sli_ks_chosen: np.array,
                             axis_label: AxisLabel,
                             path: str):
    fig, ax = plt.subplots()
    fig.set_size_inches(FigSize.DEFAULT.value)

    labels_per_choice = []
    for index, choice in enumerate(choice_method()):
        labels_per_choice.append(f'ACC k ={acc_ks_chosen[index]}\nSLI k={sli_ks_chosen[index]}')
    labels = [f'{choice_method()[index]}\n{label}' for index, label in enumerate(labels_per_choice)]
    x_values = np.arange(len(labels))
    width = 0.1

    # all_pairs
    plt.axhline(all_pairs_mean, linestyle='dashed', label=sources()[0].replace('_', ' ').capitalize(),
                color=colors('blue'))

    # intra_community
    intra_indexes = [0, 2, 4]
    acc_intra = acc_means[intra_indexes]
    sli_intra = sli_means[intra_indexes]

    # inter_community
    inter_indexes = [1, 3, 5]
    acc_inter = acc_means[inter_indexes]
    sli_inter = sli_means[inter_indexes]

    ax.bar(x_values - (3 * width / 2), acc_intra, width, color=colors('orange'),
           label='{} ACC'.format(sources()[1].replace('_', ' ').capitalize()))
    ax.bar(x_values - (1 * width / 2), acc_inter, width, color=colors('green'),
           label='{} ACC'.format(sources()[2].replace('_', ' ').capitalize()))
    ax.bar(x_values + (1 * width / 2), sli_intra, width, color=colors('cyan'),
           label='{} SLI'.format(sources()[1].replace('_', ' ').capitalize()))
    ax.bar(x_values + (3 * width / 2), sli_inter, width, color=colors('purple'),
           label='{} SLI'.format(sources()[2].replace('_', ' ').capitalize()))

    ax.set_ylabel(axis_label.value, fontsize=FontSize.DEFAULT.value)
    ax.set_xlabel(AxisLabel.K.value, fontsize=FontSize.DEFAULT.value)

    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, fontsize=FontSize.DEFAULT.value)
    ax.tick_params(axis='both', which='both', labelsize=FontSize.DEFAULT.value)
    ax.legend(loc=Legend.BEST_LOCATION.value, fontsize=FontSize.DEFAULT.value)

    y_min, y_max = plt.gca().get_ylim()
    diff = y_max - y_min
    plt.ylim(bottom=max(0, y_min), top=y_max + (0.3 * diff))

    y_tick_2decimal()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_time_evolution(all_pairs: list,
                        fed_acc_intra: list,
                        fed_acc_inter: list,
                        fed_sli_intra: list,
                        fed_sli_inter: list,
                        cen_acc_intra: list,
                        cen_acc_inter: list,
                        cen_sli_intra: list,
                        cen_sli_inter: list,
                        axis_label: AxisLabel,
                        path: str):
    plt.figure(figsize=FigSize.WIDER.value)
    x_values = np.arange(1, len(all_pairs[0]) + 1)

    curves = [all_pairs,
              fed_acc_intra,
              fed_sli_intra,
              cen_acc_intra,
              cen_sli_intra,
              fed_acc_inter,
              fed_sli_inter,
              cen_acc_inter,
              cen_sli_inter]
    labels = [sources()[0].capitalize(),
              f'{sources()[1].capitalize()} - FL-based/ACC',
              f'{sources()[1].capitalize()} - FL-based/SLI',
              f'{sources()[1].capitalize()} - Centralized/ACC',
              f'{sources()[1].capitalize()} - Centralized/SLI',
              f'{sources()[2].capitalize()} - FL-based/ACC',
              f'{sources()[2].capitalize()} - FL-based/SLI',
              f'{sources()[2].capitalize()} - Centralized/ACC',
              f'{sources()[2].capitalize()} - Centralized/SLI']

    for index, curve in enumerate(curves):
        lower_bounds = curve[0]
        means = curve[1]
        upper_bounds = curve[2]

        fill_between(x_values, lower_bounds, means, upper_bounds, labels[index])

    y_min, y_max = plt.gca().get_ylim()
    diff = y_max - y_min
    plt.ylim(bottom=max(0, y_min), top=y_max+(0.3*diff))
    plt.ylabel(axis_label.value, fontsize=FontSize.DEFAULT.value)
    plt.xlabel(AxisLabel.INTERVAL.value, fontsize=FontSize.DEFAULT.value)
    plt.xticks(x_values, fontsize=FontSize.DEFAULT.value)
    plt.legend(loc=Legend.BEST_LOCATION.value, ncol=Legend.N_COLUMNS_2.value, fontsize=FontSize.DEFAULT.value)

    y_tick_2decimal()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_losses_hyperparameters(path: str):
    files = [file for file in sorted(os.listdir(path), reverse=True) if file.endswith('csv')]

    plt.figure(figsize=FigSize.SQUARE.value)

    minimum = 1
    maximum = 0

    for file in files:
        label = file.replace('.csv', '').replace(' losses', '')
        data = np.loadtxt(f'{path}/{file}', delimiter=',')

        if data.min() < minimum:
            minimum = data.min()
        if data.max() > maximum:
            maximum = data.max()

        steps = int(len(data) / 10) if len(data) >= 50 else 5
        if len(data) <= 10:
            steps = len(data)

        x_values = np.arange(1, len(data) + 1)
        plt.xticks(np.arange(0, len(data) + 1, steps), fontsize=FontSize.DEFAULT.value)

        plt.plot(x_values, data, linewidth=2.5, label=label)

    minimum = max(0.0, minimum - 0.01)
    maximum = maximum + 0.01

    plt.ylim([minimum, maximum])
    plt.yticks(np.arange(minimum, maximum, 0.01), fontsize=FontSize.DEFAULT.value)
    plt.ylabel(AxisLabel.LOSS.value, fontsize=FontSize.DEFAULT.value)

    plt.legend(loc=Legend.BEST_LOCATION.value, ncol=Legend.N_COLUMNS_2.value, fontsize=FontSize.DEFAULT.value)

    plt.xlabel(AxisLabel.ROUND.value, fontsize=FontSize.DEFAULT.value)

    y_tick_2decimal()

    plt.tight_layout()
    plt.savefig(f'{path}/losses hyperparameter.png')
    plt.close()


def plot_avg_correlations(pearson_means, pearson_ci, spearman_means, spearman_ci, kendal_means, kendal_ci, path):
    def plot_type_correlation(means, ci, label, kwargs):
        lower_bounds = []
        upper_bounds = []
        for ci_tuple in ci:
            lower_bounds.append(ci_tuple[0])
            upper_bounds.append(ci_tuple[1])
        fill_between(intervals, lower_bounds, means, upper_bounds, label=label, kwargs=kwargs)

    plt.figure(figsize=FigSize.WIDER.value)
    intervals = np.arange(1, len(pearson_means) + 1)

    plot_type_correlation(pearson_means, pearson_ci, 'Pearson',
                          kwargs={'color': colors('blue'), 'linestyle': 'dashdot', 'linewidth': 2})
    plot_type_correlation(spearman_means, spearman_ci, 'Spearman',
                          kwargs={'color': colors('red'), 'linestyle': 'solid', 'linewidth': 2})
    plot_type_correlation(kendal_means, kendal_ci, 'Kendal',
                          kwargs={'color': colors('green'), 'linestyle': 'dashed', 'linewidth': 2})

    plt.ylabel(AxisLabel.MEAN_CORRELATION.value, fontsize=FontSize.DEFAULT.value)
    plt.xlabel(AxisLabel.INTERVAL.value, fontsize=FontSize.DEFAULT.value)
    plt.xticks(intervals, fontsize=FontSize.DEFAULT.value)
    plt.legend(loc=Legend.BEST_LOCATION.value, ncol=Legend.N_COLUMNS_3.value, fontsize=FontSize.DEFAULT.value)

    y_tick_2decimal()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_correlations(pearson_matrix, pearson_mean, spearman_matrix, spearman_mean, kendal_matrix, kendal_mean, path):
    plt.figure(figsize=FigSize.WIDER.value)
    pairs_index = np.arange(1, len(pearson_matrix.flatten()) + 1)

    plt.scatter(pairs_index, pearson_matrix.flatten(), s=0.75, color=colors('blue'))
    plt.scatter(pairs_index, spearman_matrix.flatten(), s=0.75, color=colors('red'))
    plt.scatter(pairs_index, kendal_matrix.flatten(), s=0.72, color=colors('green'))

    plt.axhline(pearson_mean, color=colors('blue'), label='Pearson', linestyle='dashdot', linewidth=2)
    plt.axhline(spearman_mean, color=colors('red'), label='Spearman', linestyle='solid', linewidth=2)
    plt.axhline(kendal_mean, color=colors('green'), label='Kendal', linestyle='dashed', linewidth=2)

    plt.ylim(top=1)
    plt.ylabel(AxisLabel.CORRELATION.value, fontsize=FontSize.DEFAULT.value)
    plt.xlabel(AxisLabel.PAIR_INDEX.value, fontsize=FontSize.DEFAULT.value)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend(loc=Legend.BEST_LOCATION.value, ncol=Legend.N_COLUMNS_3.value, fontsize=FontSize.DEFAULT.value)

    y_tick_2decimal()

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def heatmap_matrix_correlation(matrix_pearson: np.array, matrix_spearman: np.array, matrix_kendal: np.array, path):
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    im1 = axs[0].imshow(matrix_pearson, cmap='viridis', interpolation='nearest')
    axs[0].set_title('Correlation Matrix - Pearson')
    fig.colorbar(im1, ax=axs[0], label='Correlation')

    im2 = axs[1].imshow(matrix_spearman, cmap='viridis', interpolation='nearest')
    axs[1].set_title('Correlation Matrix - Spearman')
    fig.colorbar(im2, ax=axs[1], label='Correlation')

    im3 = axs[2].imshow(matrix_kendal, cmap='viridis', interpolation='nearest')
    axs[2].set_title('Correlation Matrix - Kendall-Tau')
    fig.colorbar(im3, ax=axs[2], label='Correlation')

    plt.tight_layout()
    plt.savefig(path)
    plt.close()
