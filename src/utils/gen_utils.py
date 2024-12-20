import matplotlib.pyplot as plt
import seaborn as sns

def line_plotting(x, y, title, x_label, y_label):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"plots/{title}.png")

def sns_line_plotting(df, x='Stage',y='Value',hue='Metrics', title = "Metric Performance: Before vs After", grid=True, axhline=-1, filepath='output/plots/sns_line_plotting_noname.png'):
    # Melt the DataFrame to long format for seaborn
    df_melted = df.melt(id_vars=hue, var_name=x, value_name=y)
    
    # Plot the line chart
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_melted, x=x, y=y, hue=hue, marker="o")

    # Set the plot title and labels
    plt.title(title)
    if axhline != -1:
        plt.axhline(axhline, color='red', linestyle='--')
        if axhline == 0:
            plt.ylim(-1, 1)
        elif axhline == 1:
            plt.ylim(0, 1.1)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend(title=hue)
    plt.grid(grid)
    plt.savefig(filepath)
    plt.close()

def sns_bar_plotting(df, x='Dataset_Type', y='Count', hue='Gender_Income', title="Bar Plot", filepath='output/plots/sns_bar_plotting_noname.png'):
    sns.barplot(data=df, x=x, y=y, hue=hue, palette='rocket')
    plt.title(title)
    plt.xlabel("")
    plt.ylabel("Proportion")
    plt.xticks(rotation=0)
    plt.legend(title=hue)

    # Save and show the plot
    plt.savefig(filepath)

def plot_fn_metrics(df_metrics, title, filepath):

    df_metrics_melted = df_metrics.melt(id_vars='Metric', var_name='Stage', value_name='Value')
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_metrics_melted, x='Metric', y='Value', hue='Stage', palette='viridis')
    plt.axhline(0, color='red', linestyle='--')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=0, ha='right')
    plt.ylim(-1, 1)
    plt.title(title)
    plt.legend(title='Metric')
    plt.grid(True)
    plt.savefig(filepath)