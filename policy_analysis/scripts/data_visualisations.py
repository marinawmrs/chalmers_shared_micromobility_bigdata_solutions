import matplotlib.pyplot as plt


def vis_dataset_overview(df, before_start_date, after_end_date):
    """
    Creates 2x2 grid of high-level dataset overview
    :param df: cleaned dataframe
    :param before_start_date: overall start date
    :param after_end_date: overall end date
    :return: None
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
