import os

import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
from scipy import interpolate, stats

from inner_functions.names import sources, choice_method
from inner_functions.path import build_path
from inner_types.learning import LearningApproach
from inner_types.names import ExportedFiles
from inner_types.plots import FigSize, AxisLabel, FontSize, Legend

matplotlib.use('Agg')


def colors(color_name: str):
    return mcolors.TABLEAU_COLORS.get(f'tab:{color_name}')


def y_tick_2decimal(font_size=FontSize.DEFAULT.value, n_decimal=2):
    locs, labels = plt.yticks()
    y_ticks = np.array([])
    format_string = '{:.' + str(n_decimal) + 'f}'
    for idx in np.linspace(locs[0], locs[-1], num=5):
        y_ticks = np.append(y_ticks, format_string.format(idx))
    plt.yticks(y_ticks.astype(float), fontsize=font_size, weight=FontSize.WEIGHT_SBOLD.value)


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

    plt.plot(x, means, linewidth=2, label=label.replace('_', ' '), **kwargs)
    plt.fill_between(x, lower_bounds, upper_bounds, alpha=0.2)


def plot_losses(training_losses: np.array, testing_losses: np.array, approach: LearningApproach, path: str):
    plt.figure(figsize=FigSize.SMALL.value)

    plt.ylabel(AxisLabel.LOSS.value, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)

    steps = int(len(training_losses) / 10) if len(training_losses) >= 50 else int(len(training_losses) / 4)
    if len(training_losses) <= 10:
        steps = len(training_losses)

    x_values = np.arange(1, len(training_losses) + 1)
    plt.xticks(np.arange(0, len(training_losses) + 1, steps), fontsize=FontSize.LARGE.value,
               weight=FontSize.WEIGHT_SBOLD.value)

    plt.plot(x_values, training_losses, '-', linewidth=2, label='Training losses')
    plt.plot(x_values, testing_losses, '--', linewidth=2, label='Testing losses')
    plt.legend(loc=Legend.BEST_LOCATION.value, ncol=Legend.N_COLUMNS_1.value,
               prop={'weight': FontSize.WEIGHT_SBOLD.value, 'size': FontSize.LARGE.value})

    if approach == LearningApproach.CEN:
        plt.xlabel(AxisLabel.EPOCH.value, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)
    else:
        plt.xlabel(AxisLabel.ROUND.value, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)

    y_min, y_max = plt.gca().get_ylim()
    plt.ylim(bottom=max(0, y_min), top=min(0.2, y_max))
    y_tick_2decimal(font_size=FontSize.LARGE.value, n_decimal=3)

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()


def plot_metric(dataframe: pd.DataFrame,
                k_candidates: np.arange,
                ks_chosen: np.array,
                axis_label: AxisLabel,
                path: str):
    plt.figure(figsize=FigSize.SQUARE_2.value)
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

    lower_bounds = np.array(lower_bounds)
    means = np.array(means)
    upper_bounds = np.array(upper_bounds)

    fill_between(k_candidates, lower_bounds, means, upper_bounds, all_pairs.capitalize())

    # intra_community curve
    intra_community = sources()[1]
    lower_bounds = dataframe.iloc[0][intra_columns].to_numpy().astype(float)
    means = dataframe.iloc[1][intra_columns].to_numpy().astype(float)
    upper_bounds = dataframe.iloc[2][intra_columns].to_numpy().astype(float)

    vertical_align = 'center'
    if 'MSE' in axis_label.value:
        vertical_align = 'bottom'

    plt.errorbar(k_candidates, means, yerr=[means - lower_bounds, upper_bounds - means],
                 fmt='^', linewidth=2, capsize=6, capthick=2, label=intra_community.capitalize())

    def plot_text(index, k, label, v_align, h_align):
        plt.text(k_candidates[index], means[index], f'{label} (k={int(k)})', rotation=270, verticalalignment=v_align,
                 horizontalalignment=h_align, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)

    if index_aic_choice == index_bic_choice == index_best_choice:
        plot_text(index_aic_choice, ks_chosen[0], 'AIC/BIC/\nBest', 'center', 'center')
    elif index_aic_choice == index_bic_choice != index_best_choice:
        plot_text(index_aic_choice, ks_chosen[0], 'AIC/BIC\n', 'center', 'center')
        plot_text(index_best_choice, ks_chosen[2], 'Best', vertical_align, 'right')
    elif index_aic_choice == index_best_choice != index_bic_choice:
        plot_text(index_aic_choice, ks_chosen[0], 'AIC/Best\n', 'center', 'center')
        plot_text(index_bic_choice, ks_chosen[1], 'BIC', vertical_align, 'right')
    elif index_aic_choice != index_bic_choice == index_best_choice:
        plot_text(index_bic_choice, ks_chosen[1], 'BIC/Best\n', 'center', 'center')
        plot_text(index_aic_choice, ks_chosen[0], 'AIC', vertical_align, 'right')
    else:
        plot_text(index_aic_choice, ks_chosen[0], 'AIC', vertical_align, 'left')
        plot_text(index_bic_choice, ks_chosen[1], 'BIC', vertical_align, 'right')
        plot_text(index_best_choice, ks_chosen[2], 'Best', vertical_align, 'right')

    # inter_community curve
    inter_community = sources()[2]
    lower_bounds = dataframe.iloc[0][inter_columns].to_numpy().astype(float)
    means = dataframe.iloc[1][inter_columns].to_numpy().astype(float)
    upper_bounds = dataframe.iloc[2][inter_columns].to_numpy().astype(float)

    plt.errorbar(k_candidates, means, yerr=[means - lower_bounds, upper_bounds - means],
                 fmt='v', linewidth=2, capsize=6, capthick=2, label=inter_community.capitalize())

    plt.ylabel(axis_label.value, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)
    plt.xlabel(AxisLabel.K.value, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)
    plt.xticks(k_candidates[::2], fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)
    plt.yticks(fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)
    plt.legend(loc=Legend.UPPER_RIGHT.value, ncol=Legend.N_COLUMNS_3.value,
               prop={'weight': FontSize.WEIGHT_SBOLD.value, 'size': FontSize.LARGE.value})

    y_min, y_max = plt.gca().get_ylim()
    diff = y_max - y_min
    plt.ylim(bottom=max(0, y_min), top=y_max + (0.2 * diff))
    y_tick_2decimal(font_size=FontSize.LARGE.value)

    plt.tight_layout()
    plt.savefig(path, dpi=400)
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

    ax.set_ylabel(axis_label.value, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)
    ax.set_xlabel(AxisLabel.K.value, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)

    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, fontsize=FontSize.LARGE.value)
    ax.tick_params(axis='both', which='both', labelsize=FontSize.LARGE.value)
    ax.legend(loc=Legend.BEST_LOCATION.value,
              prop={'weight': FontSize.WEIGHT_SBOLD.value, 'size': FontSize.LARGE.value})

    y_min, y_max = plt.gca().get_ylim()
    diff = y_max - y_min
    plt.ylim(bottom=max(0, y_min), top=y_max + (0.5 * diff))
    plt.xticks(fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)
    plt.yticks(fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)

    y_tick_2decimal(font_size=FontSize.LARGE.value)

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()


def plot_time_evolution(curve: list, axis_label: AxisLabel, label: str, line_style: str,
                        initial: bool = True, final: bool = False, path: str = None):
    if initial:
        plt.figure(figsize=FigSize.WIDER.value)
    x_values = np.arange(0, len(curve[0]))

    lower_bounds = curve[0]
    means = curve[1]
    upper_bounds = curve[2]

    if len(lower_bounds) > 3 and len(means) > 3 and len(upper_bounds) > 3:
        fill_between(x_values, lower_bounds, means, upper_bounds, label, kwargs={'ls': line_style})

    if final:
        y_min, y_max = plt.gca().get_ylim()
        diff = y_max - y_min
        plt.ylim(bottom=max(0, y_min), top=y_max + (0.5 * diff))
        plt.ylabel(axis_label.value, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)
        plt.xlabel(AxisLabel.INTERVAL.value, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)
        plt.xticks(np.arange(0, len(curve[0]), 2), fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)
        plt.legend(loc=Legend.BEST_LOCATION.value, ncol=Legend.N_COLUMNS_2.value,
                   prop={'weight': FontSize.WEIGHT_SBOLD.value, 'size': FontSize.DEFAULT.value})

        y_tick_2decimal(font_size=FontSize.LARGE.value)

        plt.tight_layout()
        plt.savefig(path, dpi=400)
        plt.close()


def plot_losses_hyperparameters(path: str, file_name: str):
    files = [file for file in sorted(os.listdir(path), reverse=True) if file.endswith('csv')]

    plt.figure(figsize=FigSize.SQUARE_2.value)

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
        plt.xticks(np.arange(0, len(data) + 1, steps), fontsize=FontSize.LARGE.value,
                   weight=FontSize.WEIGHT_SBOLD.value)

        plt.plot(x_values, data, linewidth=2.5, label=label)

    minimum = max(0.0, minimum - 0.01)
    maximum = maximum + 0.01

    plt.ylim([minimum, maximum])
    plt.yticks(np.arange(minimum, maximum, 0.01), weight=FontSize.WEIGHT_SBOLD.value)
    plt.ylabel(AxisLabel.LOSS.value, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)

    plt.legend(loc=Legend.BEST_LOCATION.value, ncol=Legend.N_COLUMNS_2.value,
               prop={'weight': FontSize.WEIGHT_SBOLD.value, 'size': FontSize.LARGE.value})

    plt.xlabel(AxisLabel.ROUND.value, fontsize=FontSize.LARGE.value, weight=FontSize.WEIGHT_SBOLD.value)

    y_tick_2decimal(font_size=FontSize.LARGE.value)

    plt.tight_layout()
    plt.savefig(f'{path}/{file_name}.pdf', dpi=400)
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
    intervals = np.arange(0, len(pearson_means))

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
    plt.savefig(path, dpi=400)
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

    plt.ylim(bottom=0, top=1)
    plt.ylabel(AxisLabel.CORRELATION.value, fontsize=FontSize.DEFAULT.value)
    plt.xlabel(AxisLabel.PAIR_INDEX.value, fontsize=FontSize.DEFAULT.value)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.legend(loc=Legend.BEST_LOCATION.value, ncol=Legend.N_COLUMNS_3.value, fontsize=FontSize.DEFAULT.value)

    y_tick_2decimal()

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()


def heatmap_matrix_correlation(matrix_pearson: np.array, matrix_spearman: np.array, matrix_kendal: np.array, path):
    fig, axs = plt.subplots(1, 3, figsize=(14, 6))

    im1 = axs[0].imshow(matrix_pearson, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    axs[0].set_title('Correlation Matrix - Pearson')
    fig.colorbar(im1, ax=axs[0], label='Correlation')

    im2 = axs[1].imshow(matrix_spearman, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    axs[1].set_title('Correlation Matrix - Spearman')
    fig.colorbar(im2, ax=axs[1], label='Correlation')

    im3 = axs[2].imshow(matrix_kendal, cmap='viridis', interpolation='nearest', vmin=0, vmax=1)
    axs[2].set_title('Correlation Matrix - Kendall-Tau')
    fig.colorbar(im3, ax=axs[2], label='Correlation')

    plt.tight_layout()
    plt.savefig(path, dpi=400)
    plt.close()


def plot_existent_result(csv_path, results_path, k_candidates, png_name, axis_label):
    curve_dataframe = pd.read_csv(csv_path, sep=',')
    ks_chosen = np.loadtxt(build_path(results_path, ExportedFiles.KS_CHOSEN.value))
    png_path = build_path(results_path, png_name)
    plot_metric(curve_dataframe, k_candidates, ks_chosen, axis_label, png_path)


def plot_opportunistic_routing_metric(metric: dict, x_ticks: list, title: str, path: str):
    plt.figure(figsize=(5, 6))
    for router_name, router_dict in metric.items():
        means = []
        lower_bounds = []
        upper_bounds = []
        for load_name, load_list in router_dict.items():
            mean = np.mean(load_list)
            conf_interval = stats.norm.interval(0.95, loc=mean, scale=stats.sem(load_list))
            means.append(mean)
            lower_bounds.append(conf_interval[0])
            upper_bounds.append(conf_interval[1])
        lower_bounds = np.array(lower_bounds)
        upper_bounds = np.array(upper_bounds)
        x = [i for i in range(len(x_ticks))]
        plt.errorbar(x, means, yerr=[means-lower_bounds, upper_bounds-means], label=router_name,
                      linewidth=2, capsize=6, capthick=2)
        last_line = plt.gca().get_lines()[-1]
        last_color = last_line.get_color()
        for index, value in enumerate(means):
            plt.text(x[index], value, f'{round(value, 2)}', color=last_color)
        plt.xticks(x, x_ticks)
    plt.xlabel('TTL (minutes)')
    plt.ylabel(title)
    plt.suptitle(title)
    plt.legend()
    plt.savefig(build_path(path, f'{title}.png'))
    plt.savefig(build_path(path, f'{title}.pdf'))
    plt.close()
