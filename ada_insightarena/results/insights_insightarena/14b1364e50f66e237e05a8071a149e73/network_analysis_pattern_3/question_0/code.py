import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assume df is already loaded as a pandas DataFrame
# Data Preparation & Cleaning
try:
    # Convert 'Made SLA' to boolean if not already
    df['Made SLA'] = df['Made SLA'].astype(bool)
    
    # Filter data for high impact incidents
    high_impact_df = df[df['Impact'] == '1 - High']
    
    # Calculate the percentage of high impact incidents that met SLA
    total_high_impact = len(high_impact_df)
    met_sla_high_impact = high_impact_df['Made SLA'].sum()
    percentage_met_sla = (met_sla_high_impact / total_high_impact) * 100 if total_high_impact > 0 else 0

except Exception as e:
    raise RuntimeError(f"Error during data preparation and cleaning: {e}")

# Visualization
try:
    # Create a bar plot to visualize the percentage of high impact incidents meeting SLA
    plt.figure(figsize=(8, 6))
    sns.barplot(x=['Met SLA', 'Did Not Meet SLA'], 
                y=[met_sla_high_impact, total_high_impact - met_sla_high_impact], 
                palette='viridis')
    plt.title('High Impact Incidents Meeting SLA')
    plt.ylabel('Number of Incidents')
    plt.xlabel('SLA Status')
    plt.ylim(0, total_high_impact + 10)  # Add some space above the bars for better visualization

    # Save the plot
    plt.savefig('insightarena/results/insights_insightarena/14b1364e50f66e237e05a8071a149e73/network_analysis_pattern_3/question_0/plot.jpeg')

except Exception as e:
    raise RuntimeError(f"Error during visualization: {e}")

# Compute & Store Key Statistics
try:
    stats = {
        'total_high_impact_incidents': total_high_impact,
        'met_sla_high_impact': met_sla_high_impact,
        'percentage_met_sla': percentage_met_sla
    }

    # Print the stats dictionary
    print(stats)

except Exception as e:
    raise RuntimeError(f"Error during statistics computation: {e}")
