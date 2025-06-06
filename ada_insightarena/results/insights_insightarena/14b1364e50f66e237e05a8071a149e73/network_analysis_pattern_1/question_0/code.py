import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assume df is already loaded as per the input details
# Data Preparation & Cleaning
try:
    # Convert 'Resolved' to datetime
    df['Resolved'] = pd.to_datetime(df['Resolved'], errors='coerce')
    df['Opened'] = pd.to_datetime(df['Opened'], errors='coerce')

    # Calculate resolution time in hours
    df['Resolution Time (hours)'] = (df['Resolved'] - df['Opened']).dt.total_seconds() / 3600

    # Drop rows with NaN resolution times
    df_clean = df.dropna(subset=['Resolution Time (hours)', 'Category', 'Priority'])

except Exception as e:
    raise RuntimeError(f"Data preparation error: {e}")

# Data Analytics Technique: Calculate average resolution times
try:
    # Group by 'Category' and 'Priority' and calculate mean resolution time
    resolution_stats = df_clean.groupby(['Category', 'Priority'])['Resolution Time (hours)'].mean().reset_index()

except Exception as e:
    raise RuntimeError(f"Data analytics error: {e}")

# Visualization
try:
    # Create a bar plot for average resolution times
    plt.figure(figsize=(12, 8))
    sns.barplot(data=resolution_stats, x='Category', y='Resolution Time (hours)', hue='Priority', ci=None)
    plt.title('Average Resolution Times by Category and Priority')
    plt.xlabel('Category')
    plt.ylabel('Average Resolution Time (hours)')
    plt.xticks(rotation=45)
    plt.legend(title='Priority')
    plt.tight_layout()

    # Save the plot
    plt.savefig('insightarena/results/insights_insightarena/14b1364e50f66e237e05a8071a149e73/network_analysis_pattern_1/question_0/plot.jpeg')

except Exception as e:
    raise RuntimeError(f"Visualization error: {e}")

# Compute & Store Key Statistics
try:
    # Create a dictionary to store key statistics
    stats = {
        'average_resolution_times': resolution_stats.to_dict(orient='list'),
        'total_categories': df_clean['Category'].nunique(),
        'total_priorities': df_clean['Priority'].nunique(),
        'total_incidents': len(df_clean)
    }

    # Print the stats dictionary
    print(stats)

except Exception as e:
    raise RuntimeError(f"Statistics computation error: {e}")
