import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.core.display import display_html
from scipy import stats
import contextily as ctx
from matplotlib import colors
import matplotlib.cm as cm


# =============================================================================
# Dataset Overview
# =============================================================================


def vis_dataset_overview(df, before_start_date, after_end_date):
    """
    Creates 2x2 grid of high-level dataset overview
    @param df: cleaned dataframe
    @param before_start_date: overall start date
    @param after_end_date: overall end date
    @return: None
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1
    x_data1 = [df[df.type == op].isodate for op in df.type.unique()]
    axs[0, 0].hist(x_data1, bins=(after_end_date - before_start_date).days, stacked=True, label=df.type.unique(),
                   width=0.8)
    axs[0, 0].legend()
    axs[0, 0].set_title('Number of recorded trips per day, by operator')
    axs[0, 0].set_xlabel('Date')
    axs[0, 0].set_ylabel('Number of Trips')
    axs[0, 0].tick_params(axis='x', rotation=60)

    # Plot 2
    unique_df = df.drop_duplicates(subset=['isodate', 'id'])
    x_data2 = [unique_df[unique_df.type == op].isodate for op in unique_df.type.unique()]
    axs[0, 1].hist(x_data2, bins=(after_end_date - before_start_date).days, stacked=True, label=unique_df.type.unique(),
                   width=0.8)
    axs[0, 1].legend()
    axs[0, 1].set_title('Number of unique vehicles per day, by operator')
    axs[0, 1].set_xlabel('Date')
    axs[0, 1].set_ylabel('Number of Vehicles')
    axs[0, 1].tick_params(axis='x', rotation=60)

    # Plot 3
    x_data3 = [df[df.type == op].day for op in df.type.unique()]
    axs[1, 0].hist(x_data3, bins=7, stacked=True, label=df.type.unique(), width=0.8)
    axs[1, 0].legend()
    axs[1, 0].set_title('Number of recorded trips, aggregated by weekday, by operator')
    axs[1, 0].set_xlabel('Weekday')
    axs[1, 0].set_ylabel('Number of Trips')
    axs[1, 0].tick_params(axis='x', rotation=60)

    # Plot 4
    axs[1, 1].pie(df.type.value_counts(), labels=df.type.value_counts().index, autopct='%1.1f%%')
    axs[1, 1].set_title('Distribution of Operators in Dataset')

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()


def aggregate_and_describe(df_before, df_after, groupby_column, dim, aggregation_function, title, ttest, ttest_log):
    """
    Aggregates the DataFrame using the specified aggregation function and describe the result

    @param df_before: pandas DataFrame
    @param df_after: pandas DataFrame
    @param groupby_column: str, column to group by
    @param dim: str, column to aggregate on
    @param aggregation_function: function, aggregation function to apply (e.g., pd.Series.nunique)
    @param title: str, title for the output DataFrame
    @param ttest: bool, whether to include T-Test or not
    @param ttest_log: bool, whether to apply log transform before t-test
    @return: pandas.io.formats.style.Styler, styled DataFrame
    """
    if aggregation_function is None:
        agg_before = pd.DataFrame(df_before[dim])
        agg_after = pd.DataFrame(df_after[dim])
    else:
        agg_before = pd.DataFrame(df_before.groupby(groupby_column)[dim].agg(aggregation_function))
        agg_after = pd.DataFrame(df_after.groupby(groupby_column)[dim].agg(aggregation_function))

    stats_before = pd.DataFrame(agg_before.describe()).rename(columns={dim: 'before'})
    stats_after = pd.DataFrame(agg_after.describe()).rename(columns={dim: 'after'})

    if ttest:
        if ttest_log:
            t_stat, p_val = stats.ttest_ind(np.log(agg_before[dim].dropna()), np.log(agg_after[dim].dropna()),
                                            equal_var=True)
        else:
            t_stat, p_val = stats.ttest_ind(agg_before[dim].dropna(), agg_after[dim].dropna(), equal_var=True)
        title += f'\nt_stat={t_stat:.4f} \n p_val={p_val:.4f}'

    stats_table = pd.concat([stats_before, stats_after], axis=1)
    stats_table = stats_table.style.set_caption(title).format(precision=4).set_table_attributes(
        "style='display:inline'")

    return stats_table


# =============================================================================
# Histograms/PDFs
# =============================================================================


def histogram_lines(data_before, data_after, e, bins, use_log=False):
    """
    Generates histogram data for a given feature

    @param data_before: DataFrame containing data before the change.
    @param data_after: DataFrame containing data after the change.
    @param e: Column to generate histogram for.
    @param bins: Number of bins for the histogram.
    @param use_log: Whether to apply logarithmic transformation to the data.
    @return: Tuple containing bin centers, histogram data before, and histogram data after.
    """
    if use_log:
        data_before = np.log(data_before[e].dropna())
        data_after = np.log(data_after[e].dropna())

    hist_before, bin_edges_before = np.histogram(data_before, bins, density=True)
    hist_after, bin_edges_after = np.histogram(data_after, bins, density=True)

    bin_centers_before = (bin_edges_before[1:] + bin_edges_before[:-1]) / 2
    bin_centers_after = (bin_edges_after[1:] + bin_edges_after[:-1]) / 2

    return bin_centers_before, bin_centers_after, hist_before, hist_after


def plot_pdf_day_mode(dims, log_arr, bins, weekdays_before, weekends_before, fri_before, weekdays_after, weekends_after,
                      fri_after):
    """
    Plot probability density functions (PDFs) for weekdays, weekends, and Fridays

    @param dims: List of dimensions/columns to plot.
    @param log_arr: List indicating whether to apply logarithmic transformation to each dimension.
    @param bins: Number of bins for the histograms.
    @param weekdays_before: DataFrame containing data for weekdays before the change.
    @param weekends_before: DataFrame containing data for weekends before the change.
    @param fri_before: DataFrame containing data for Fridays before the change.
    @param weekdays_after: DataFrame containing data for weekdays after the change.
    @param weekends_after: DataFrame containing data for weekends after the change.
    @param fri_after: DataFrame containing data for Fridays after the change.
    """
    fig, axes = plt.subplots(len(dims), 3, figsize=(13, 3 * len(dims)))

    for i, e in enumerate(dims):
        arr_bef = [weekdays_before, fri_before, weekends_before]
        arr_aft = [weekdays_after, fri_after, weekends_after]

        for j in range(3):
            if log_arr[i] == 1:
                plot_weekdays_before = np.histogram(np.log(arr_bef[j][e].dropna()), bins, density=True)
                plot_weekdays_after = np.histogram(np.log(arr_aft[j][e].dropna()), bins, density=True)
            else:
                plot_weekdays_before = np.histogram(arr_bef[j][e].dropna(), bins, density=True)
                plot_weekdays_after = np.histogram(arr_aft[j][e].dropna(), bins, density=True)

            plot_weekdays_before_l = (plot_weekdays_before[1][1:] + plot_weekdays_before[1][:-1]) / 2
            plot_weekdays_after_l = (plot_weekdays_after[1][1:] + plot_weekdays_after[1][:-1]) / 2

            if len(dims) == 1:
                axes[j].plot(plot_weekdays_before_l, plot_weekdays_before[0], label='ant')
                axes[j].plot(plot_weekdays_after_l, plot_weekdays_after[0], label='post', alpha=0.7)
                axes[j].set_xlabel(e)
                axes[j].set_ylabel('Probability')
                axes[j].legend()
                axes[0].set_title('Weekdays (Mon-Thur)')
                axes[1].set_title('Fridays')
                axes[2].set_title('Weekends (Sat-Sun)')
            else:
                axes[i][j].plot(plot_weekdays_before_l, plot_weekdays_before[0], label='ant')
                axes[i][j].plot(plot_weekdays_after_l, plot_weekdays_after[0], label='post', alpha=0.7)
                axes[i][j].set_xlabel(e)
                axes[i][j].set_ylabel('Probability')
                axes[i][j].legend()
                axes[0][0].set_title('Weekdays (Mon-Thur)')
                axes[0][1].set_title('Fridays')
                axes[0][2].set_title('Weekends (Sat-Sun)')
    plt.tight_layout()
    plt.show()


def plot_demand_pdf_throughout_weekdays(density, weekdays_before, weekends_before, fri_before, weekdays_after,
                                        weekends_after, fri_after):
    """

    @param density: bool, whether to plot histogram (absolute) or PDF
    @param weekdays_before: DataFrame containing data for weekdays before the change.
    @param weekends_before: DataFrame containing data for weekends before the change.
    @param fri_before: DataFrame containing data for Fridays before the change.
    @param weekdays_after: DataFrame containing data for weekdays after the change.
    @param weekends_after: DataFrame containing data for weekends after the change.
    @param fri_after: DataFrame containing data for Fridays after the change.
    @param return:
    """
    arr_bef = [weekdays_before, fri_before, weekends_before]
    arr_aft = [weekdays_after, fri_after, weekends_after]
    labels = ['Weekdays (Mon-Thurs)', 'Fridays', 'Weekends']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for i, e in enumerate(arr_bef):
        plot_xdays_before = np.histogram(arr_bef[i].o_time.dt.hour, 24, density=density)
        plot_xdays_after = np.histogram(arr_aft[i].o_time.dt.hour, 24, density=density)

        axes[i].plot(plot_xdays_before[1][:-1], plot_xdays_before[0], label='ant')
        axes[i].plot(plot_xdays_after[1][:-1], plot_xdays_after[0], label='post')
        axes[i].set_xlabel('Hour of the day')
        axes[i].set_ylabel('Probability (Trip)' if density else 'Number of Trips')
        axes[i].set_title(labels[i])
        axes[i].legend()

    plt.tight_layout()
    plt.show()


def throughout_weekdays_plot(dimensions, weekdays_before, fri_before, weekends_before, weekdays_after, fri_after,
                             weekends_after):
    """
     Plots the mean and standard deviation of the given variable throughout the hours of the day

    @param dimensions: List of dimensions/columns to plot.
    @param weekdays_before: DataFrame containing data for weekdays before the change.
    @param weekends_before: DataFrame containing data for weekends before the change.
    @param fri_before: DataFrame containing data for Fridays before the change.
    @param weekdays_after: DataFrame containing data for weekdays after the change.
    @param weekends_after: DataFrame containing data for weekends after the change.
    @param fri_after: DataFrame containing data for Fridays after the change.
    @return:
    """

    labels = ['Weekdays (Mon-Thurs)', 'Fridays', 'Weekends']
    fig, axes = plt.subplots(len(dimensions), 3, figsize=(13, 3 * len(dimensions)))
    arr_bef = [weekdays_before, fri_before, weekends_before]
    arr_aft = [weekdays_after, fri_after, weekends_after]

    for i, dim in enumerate(dimensions):
        for j in range(3):
            # Calculate mean and standard deviation for each hour of the day
            arr_bef[j] = arr_bef[j].assign(hours=lambda x: x.o_time.dt.hour)
            mean_bef = arr_bef[j].groupby('hours')[dim].mean()
            std_bef = arr_bef[j].groupby('hours')[dim].std()

            arr_aft[j] = arr_aft[j].assign(hours=lambda x: x.o_time.dt.hour)
            mean_aft = arr_aft[j].groupby('hours')[dim].mean()
            std_aft = arr_aft[j].groupby('hours')[dim].std()

            if len(dimensions) == 1:
                ax = axes[j]
            else:
                ax = axes[i][j]

            ax.plot(mean_bef.reset_index().hours, mean_bef, label='ant')
            ax.fill_between(mean_bef.reset_index().hours, mean_bef - (0.5 * std_bef), mean_bef + (0.5 * std_bef),
                            alpha=0.2)

            ax.plot(mean_aft.reset_index().hours, mean_aft, label='post')
            ax.fill_between(mean_aft.reset_index().hours, mean_aft - (0.5 * std_aft), mean_aft + (0.5 * std_aft),
                            alpha=0.2)

            ax.legend()
            ax.set_xlabel('Mean ' + dim + ' per hour of day')

            if i == 0:
                ax.set_ylabel('seconds')
            elif i == 1:
                ax.set_ylabel('meters')
            elif i == 2:
                ax.set_ylabel('m/s')

            ax.set_title(labels[j])

    plt.tight_layout()
    plt.show()


def plot_scooter_efficiency_histogram(data_before, data_after, types):
    """
    Creates PDFs of usage efficiency and number of scooters

    @param data_before: DataFrame containing data before the change.
    @param data_after: DataFrame containing data after the change.
    @param types: array, list of operators
    @return:
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    colors = cm.Set2(np.linspace(0, 1, 8)) if len(types) > 1 else cm.tab10(np.linspace(0, 1, 10))

    for i, t in enumerate(types):
        color = colors[i % len(colors)] if len(types) > 1 else None

        # Overall utilisation rate
        plot_count_before = np.histogram(data_before[data_before['type'].isin(t)].groupby(['id']).o_time.count(), 25,
                                         density=True)
        plot_count_after = np.histogram(data_after[data_after['type'].isin(t)].groupby(['id']).o_time.count(), 25,
                                        density=True)

        plot_count_before_l = (plot_count_before[1][1:] + plot_count_before[1][:-1]) / 2
        plot_count_after_l = (plot_count_after[1][1:] + plot_count_after[1][:-1]) / 2

        if np.sum(plot_count_before[0]) != 0:
            axes[0].plot(plot_count_before_l, plot_count_before[0], label=f'{t} - ant', color=color, alpha=0.7)
        if np.sum(plot_count_after[0]) != 0:
            axes[0].plot(plot_count_after_l, plot_count_after[0], label=f'{t} - post', color=color, alpha=0.4)

        axes[0].set_ylabel('Probability')
        axes[0].set_xlabel('Trips per Scooter')
        axes[0].set_title('Total Usage Efficiency in Timeframe')
        axes[0].legend()

        # Daily utilisation rate
        plot_count_before = np.histogram(
            data_before[data_before['type'].isin(t)].groupby(['Date', 'id']).size().reset_index()[0], 14, density=True)
        plot_count_after = np.histogram(
            data_after[data_after['type'].isin(t)].groupby(['Date', 'id']).size().reset_index()[0], 12, density=True)

        plot_count_before_l = (plot_count_before[1][1:] + plot_count_before[1][:-1]) / 2
        plot_count_after_l = (plot_count_after[1][1:] + plot_count_after[1][:-1]) / 2

        if np.sum(plot_count_before[0]) != 0:
            axes[1].plot(plot_count_before_l, plot_count_before[0], label=f'{t} - ant', color=color, alpha=0.7)
        if np.sum(plot_count_after[0]) != 0:
            axes[1].plot(plot_count_after_l, plot_count_after[0], label=f'{t} - post', color=color, alpha=0.4)

        axes[1].set_ylabel('Probability')
        axes[1].set_xlabel('Trips per Scooter')
        axes[1].set_title('Daily Usage Efficiency')
        axes[1].legend()

        # Number of scooters per day
        before_counts = data_before[data_before['type'].isin(t)].groupby(['Date']).id.nunique()
        after_counts = data_after[data_after['type'].isin(t)].groupby(['Date']).id.nunique()

        if len(before_counts) > 0:
            axes[2].hist(before_counts, bins=30, density=True, width=15, alpha=0.7, label=f'{t} - ant', color=color)
        if len(after_counts) > 0:
            axes[2].hist(after_counts, bins=30, density=True, width=15, alpha=0.4, label=f'{t} - post', color=color)

        axes[2].set_ylabel('Probability')
        axes[2].set_xlabel('Scooters')
        axes[2].set_title('Daily Number of Scooters')
        axes[2].legend()

    plt.tight_layout()
    plt.show()


def plot_mean_per_scooter_histogram(data_before, data_after, dims, types):
    """
    Creates PDFs of mean distance, time, and speed per scooter

    @param data_before: DataFrame containing data before the change.
    @param data_after: DataFrame containing data after the change.
    @param dims: array, list of dimensions
    @param types: array, list of operator types to filter the data
    @return:
    """
    fig, axes = plt.subplots(1, len(dims), figsize=(18, 4))

    colors = cm.Set2(np.linspace(0, 1, 8))

    for i, e in enumerate(dims):
        for j, t in enumerate(types):
            color = colors[j % len(colors)] if len(types) > 1 else None

            # average distance, time and speed per scooter
            plot_dim_before = np.histogram(data_before[data_before['type'].isin(t)].groupby(['id'])[e].mean(), 50,
                                           density=True)
            plot_dim_after = np.histogram(data_after[data_after['type'].isin(t)].groupby(['id'])[e].mean(), 50,
                                          density=True)

            plot_dim_before_l = (plot_dim_before[1][1:] + plot_dim_before[1][:-1]) / 2
            plot_dim_after_l = (plot_dim_after[1][1:] + plot_dim_after[1][:-1]) / 2

            axes[i].plot(plot_dim_before_l, plot_dim_before[0], label=f'{t} - ant', color=color,
                         alpha=0.8 if len(types) > 1 else 1)
            axes[i].plot(plot_dim_after_l, plot_dim_after[0], label=f'{t} - post', color=color,
                         alpha=0.4 if len(types) > 1 else 1)
            axes[i].set_xlabel(e + ' by scooter')
            axes[i].set_ylabel('Probability')
            # axes[i].set_text(2, 16, 'Second Title', fontsize=12, color='red', ha='center')

            if len(types) == 1:
                mean_before = data_before[data_before['type'].isin(t)][e].mean()
                mean_after = data_after[data_after['type'].isin(t)][e].mean()
                axes[i].text(1, 0.5, f'Mean (ant): {mean_before:.2f}' + '\n' + f'Mean (post): {mean_after:.2f}',
                             horizontalalignment='right', verticalalignment='center', transform=axes[i].transAxes)
            axes[i].legend()

    plt.tight_layout()
    plt.show()


def figure_modal_subs(series_arr):
    """
    Creates stacked histogram bar-plot of modal substitution rate, per zone and trip level
    @param series_arr: array, with subarrays for trip-level and zone-level
    @return:
    """
    arr = ['before', 'after']
    modes = ['P_walk', 'P_car', 'P_taxi', 'P_bike', 'P_PT']
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    for li, level in enumerate(['trip', 'zone']):
        series = series_arr[li]
        subs_mode = pd.DataFrame()

        if level == 'zone':
            modes = [i + '_avg' for i in modes]

        for i, s in enumerate(series):
            subs_mode[arr[i]] = s[modes].idxmax(axis=1).value_counts(normalize=True)

        subs_mode = subs_mode.fillna(0)
        bottom = [0, 0]
        pad = [-1, -1]

        for row in subs_mode.iterrows():
            ax[li].bar(['before', 'after'], [row[1].before, row[1].after], label=row[0], bottom=bottom)

            for idx, val in enumerate([row[1].before, row[1].after]):
                if val < 0.01 and val > 0:
                    pad[idx] += 1
                y_pos = val / 2 + bottom[idx] + pad[idx] * 0.02
                ax[li].text(idx, y_pos, f'{val:.4f}', ha='center')

            bottom[0] += row[1].before
            bottom[1] += row[1].after

        ax[li].set_title(f'Likelihood of substituted mode, {level}-level')
        ax[li].set_xlabel('Interval')
        ax[li].set_ylabel('Rate')
        ax[li].legend()

    plt.tight_layout()
    plt.show()


def get_intercept(x1, y1, x2, y2):
    """
    Helper Function for Balance Points, computes 0-intercept

    @param x1: float
    @param y1: float
    @param x2: float
    @param y2: float
    @return:
    """
    k = (y1 - y2) / (x1 - x2)
    b = y1 - k * x1
    intercept = -b / k
    return intercept


def plot_balance_points(series):
    """
    Generates balance point plots

    @param series: array, before and after pandas dataframe
    @return:
    """
    color_list = plt.cm.tab10.colors

    fig, ax = plt.subplots(figsize=(10, 5))

    for j, s in enumerate(series):
        s['distance_round'] = np.floor(s.escooter_distance / 1000)
        for x in range(0, 10):
            sx = s[s.distance_round == x]
            mean_l = []
            for i in np.linspace(0, 60, 11):
                sx['GHG2'] = sx.car_distance / 1000 * 160.7 * (
                            sx.P_car + sx.P_taxi) + sx.transit_transitdistance / 1000 * 16.04 * sx.P_PT + sx.escooter_distance / 1000 * 37.0 * sx.P_bike - sx.escooter_distance / 1000 * i
                mean_l.append(sx.GHG2.mean())
            zero_point = round(get_intercept(0, mean_l[0], 60, mean_l[-1]), 2)
            ax.plot(np.linspace(0, 60, 11), mean_l, label=str(x) + '-' + str(x + 1) + ' km', color=color_list[j],
                    alpha=0.2 + 0.075 * x, linestyle='dashed')

        mean_l = []
        for i in np.linspace(0, 60, 11):
            s['GHG2'] = s.car_distance / 1000 * 160.7 * (
                        s.P_car + s.P_taxi) + s.transit_transitdistance / 1000 * 16.04 * s.P_PT + s.escooter_distance / 1000 * 37.0 * s.P_bike - s.escooter_distance / 1000 * i
            mean_l.append(s.GHG2.mean())
        zero_point = round(get_intercept(0, mean_l[0], 60, mean_l[-1]), 2)

        ax.plot(np.linspace(0, 60, 11), mean_l, label='Average', color=color_list[j])

        ax.plot(zero_point, 0, marker='o', markersize=6, color=color_list[j])
        print('Balance point in Timeframe', j + 1, ':', zero_point)

    ax.set_title('Balance Points')
    ax.set_xlabel('Emission factor of SES (CO2 g/km)')
    ax.set_ylabel('GHG Reduction (CO2/g)')
    ax.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()

    plt.show()


# =============================================================================
# T-Tests
# =============================================================================


def reg_ttest(dimensions, log_arr, data_before, data_after):
    """
    Performs regular t-test on the specified dimensions between data_before and data_after

    @param dimensions: List of columns to perform t-test on
    @param log_arr: List indicating whether to apply logarithmic transformation to each dimension
    @param data_before: DataFrame containing data before the change
    @param data_after: DataFrame containing data after the change
    @return: DataFrame containing t-test results
    """
    df_ttest = pd.DataFrame(index=['t-statistic', 'p_ttest'])

    for c, i in enumerate(dimensions):
        if log_arr[c] == 1:
            t_stat, p_val = stats.ttest_ind(np.log(data_before[i].dropna()), np.log(data_after[i].dropna()),
                                            equal_var=True)
        else:
            t_stat, p_val = stats.ttest_ind(data_before[i].dropna(), data_after[i].dropna(), equal_var=True)

        df_ttest[i] = [t_stat, p_val]

    # Mean and standard deviation of quantities, before and after
    df_ttest.loc['mean_ant'] = data_before[dimensions].mean()
    df_ttest.loc['mean_post'] = data_after[dimensions].mean()
    df_ttest.loc['std_ant'] = data_before[dimensions].std()
    df_ttest.loc['std_post'] = data_after[dimensions].std()

    return df_ttest


def ttest_day_mode(dims, arr_log, weekdays_before, weekends_before, fri_before, weekdays_after, weekends_after,
                   fri_after):
    """
    Performs t-tests for weekdays, weekends, and Fridays before and after changes

    @param dims: List of dimensions/columns to perform t-tests on.
    @param arr_log: List indicating whether to apply logarithmic transformation to each dimension.
    @param weekdays_before: DataFrame containing data for weekdays before the change.
    @param weekends_before: DataFrame containing data for weekends before the change.
    @param fri_before: DataFrame containing data for Fridays before the change.
    @param weekdays_after: DataFrame containing data for weekdays after the change.
    @param weekends_after: DataFrame containing data for weekends after the change.
    @param fri_after: DataFrame containing data for Fridays after the change.
    @return: DataFrame containing t-test results.
    """
    # Create DataFrames for t-test results
    df_ttest = pd.DataFrame(index=['t-statistic', 'p_ttest'], columns=dims)
    df_ttest_WE = pd.DataFrame(index=['t-statistic', 'p_ttest'], columns=dims)
    df_ttest_fri = pd.DataFrame(index=['t-statistic', 'p_ttest'], columns=dims)

    arr = [df_ttest, df_ttest_WE, df_ttest_fri]
    arr_dfs_before = [weekdays_before, weekends_before, fri_before]
    arr_dfs_after = [weekdays_after, weekends_after, fri_after]

    # Perform t-tests and populate DataFrames
    for f, j in enumerate(dims):
        for i, df_ttest in enumerate(arr):
            data_before = arr_dfs_before[i][j].dropna()
            data_after = arr_dfs_after[i][j].dropna()
            if arr_log[f] == 1:
                t_stat, p_val = stats.ttest_ind(np.log(data_before), np.log(data_after), equal_var=True)
            else:
                t_stat, p_val = stats.ttest_ind(data_before, data_after, equal_var=True)
            arr[i].loc['t-statistic', j] = t_stat
            arr[i].loc['p_ttest', j] = p_val

            # Calculate mean and standard deviation of quantities before and after
            arr[i].loc['mean_ant', j] = data_before.mean()
            arr[i].loc['mean_post', j] = data_after.mean()
            arr[i].loc['std_ant', j] = data_before.std()
            arr[i].loc['std_post', j] = data_after.std()

    # Display styled DataFrames
    scooter_stats_styler = df_ttest.style.set_table_attributes("style='display:inline'").set_caption('Weekdays').format(precision=4)
    scooter_fri_stats_styler = df_ttest_fri.style.set_table_attributes("style='display:inline'").set_caption('Fridays').format(precision=4)
    scooter_WE_stats_styler = df_ttest_WE.style.set_table_attributes("style='display:inline'").set_caption('Weekends').format(precision=4)

    display_html(scooter_stats_styler._repr_html_() + scooter_fri_stats_styler._repr_html_() +
                 scooter_WE_stats_styler._repr_html_(), raw=True)


def paired_ttest_az(dimensions, zone_aggr):
    """
    Performs a paired t-test based on pre-aggregated analysis zones

    @param dimensions: list of strings, names of dimensions to perform the t-test on
    @param zone_aggr: array, contains the before and after GeoDataframe
    @return: DataFrame containing t-statistic and p-value for each dimension, along with mean and standard deviation
             before and after aggregation
    """
    df_ttest = pd.DataFrame(columns=dimensions)

    df_ttest['measure'] = ['t-statistic', 'p_ttest']
    df_ttest.set_index('measure', inplace=True)

    intersection_zone_ids = pd.merge(zone_aggr[0], zone_aggr[1], on='id').id
    i1 = zone_aggr[0][zone_aggr[0]['id'].isin(intersection_zone_ids)]
    i2 = zone_aggr[1][zone_aggr[1]['id'].isin(intersection_zone_ids)]

    for i in dimensions:
        temp_ttest = stats.ttest_rel(i1[i], i2[i], nan_policy='omit')

        df_ttest[i] = [temp_ttest.statistic, temp_ttest.pvalue]

    df_ttest.loc['mean_ant'] = i1[dimensions].mean()
    df_ttest.loc['mean_post'] = i2[dimensions].mean()
    df_ttest.loc['std_ant'] = i1[dimensions].std()
    df_ttest.loc['std_post'] = i2[dimensions].std()

    return df_ttest.style.set_caption('Paired T-Test between Zones').format(precision=4)


# =============================================================================
# Maps
# =============================================================================

def plot_before_after_az(dimensions, zone_aggr, unit):
    """
    Plot the before and after comparison of the given dimensions across analysis zones

    @param dimensions: list of strings, names of dimensions to plot
    @param zone_aggr: array, contains the before and after GeoDataframe
    @param unit: string, unit of the variable
    @return: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
    """

    fig, axes = plt.subplots(len(dimensions), 2, figsize=(13, 4 * len(dimensions)))

    for i in range(2):
        zone_aggr[i] = zone_aggr[i].to_crs(epsg=3857)
        for j, dim in enumerate(dimensions):
            max_scale = max(zone_aggr[0][dim].max(), zone_aggr[1][dim].max())
            min_scale = min(zone_aggr[0][dim].min(), zone_aggr[1][dim].min())

            ax_index = axes[j][i] if len(dimensions) > 1 else axes[i]
            ax = zone_aggr[i].plot(ax=ax_index, column=dim, legend=True, alpha=0.6, vmax=max_scale, vmin=min_scale)
            ctx.add_basemap(ax=ax_index, crs=zone_aggr[i].crs, source=ctx.providers.CartoDB.Positron, alpha=0.5)
            ax_index.set_title(dim + (' (Pre-Policy)' if i == 0 else ' (Post-Policy)') + ' in ' + unit)
            ax.axis('off')

    return fig, axes


def violin_zone_level(ax, zone_var, unit, *series):
    """
    Creates Violin Plot for zone maps

    @param ax: plotly axis
    @param zone_var: string, current column
    @param unit: string, unti for variable
    @param series: array of GeoDataFrames
    @return:
    """

    for i, s in enumerate(series):
        zone = s[zone_var].dropna()
        r = ax.violinplot(dataset=[zone], positions=[i], showmeans=True, showmedians=True, showextrema=False, widths=0.7,
                      points=200, bw_method='scott', vert=True)
        r['cmeans'].set_linestyle('dotted')

        violin_annotation(ax, i, s, zone_var)

    ax.set_title(zone_var + ', zone-level')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['before', 'after'])
    ax.set_ylabel('{}'.format(zone_var) + ' ({}]'.format(unit))

def violin_annotation(ax, i, s, variable):
    """
    Appends annotations to violin plot

    @param ax: plotly axis
    @param i: int, current series
    @param s: GeoDataFrame, current series
    @param variable: string, current column
    @return:
    """
    pos_impact = (s[variable] > 0).sum() / len(s[variable].dropna())
    mean_val = s[variable].mean()
    median_val = s[variable].median()

    ax.annotate("Positive impact: {:.2%}".format(pos_impact), xy=(i, 1), xycoords='axes fraction', xytext=(0, 10),
                textcoords='offset points', ha='center', va='bottom', fontsize=8, color='blue' if i == 0 else 'orange')

    ax.annotate("Mean: {:.2f}".format(mean_val) + "\nMedian: {:.2f}".format(median_val), xy=(i, 0), xycoords='axes fraction', xytext=(0, -15),
                textcoords='offset points', ha='center', fontsize=8, color='blue' if i == 0 else 'orange')


def plot_differences_az(dimensions, zone_aggr, unit):
    """
    Plot the differences between post-policy and pre-policy values for the given dimensions across analysis zones.

    @param dimensions: list of strings, names of dimensions to plot differences for
    @param zone_aggr: array, contains the before and after GeoDataframe
    @param unit: string, unit of the variable
    @return: matplotlib.figure.Figure, matplotlib.axes._subplots.AxesSubplot
    """

    fig, axes = plt.subplots(len(dimensions), 2, figsize=(13, 5 * len(dimensions)))

    for j, dim in enumerate(dimensions):
        # Calculate difference for each zone
        cell_diff = zone_aggr[1].copy()
        cell_diff[dim] = cell_diff[dim] - zone_aggr[0][dim]
        cell_diff[dim] = cell_diff[dim].dropna()
        cell_diff = cell_diff.to_crs(epsg=3857)

        max_scale = max(np.abs(cell_diff[dim].min()), np.abs(cell_diff[dim].max()))
        divnorm = colors.TwoSlopeNorm(vmin=-max_scale, vcenter=0, vmax=max_scale)
        ax = cell_diff.plot(ax=axes[j][0], column=dim, legend=True, alpha=0.7, cmap='RdYlGn', norm=divnorm)
        ctx.add_basemap(ax=axes[j][0], crs=cell_diff.crs, source=ctx.providers.CartoDB.Positron, alpha=0.5)
        axes[j][0].set_title(dim + ' - Δ(Post-Pre) in ' + unit)
        ax.axis('off')

        violin_zone_level(axes[j][1],dim, unit, zone_aggr[0], zone_aggr[1])

    return fig, axes
