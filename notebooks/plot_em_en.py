import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the Seaborn style to "whitegrid" for an academic look
sns.set_style("whitegrid")

# Read the CSV file
df = pd.read_csv(os.path.join("..", "src", "emissions.csv"))


# Group by 'project_name' and sum the values for each project
grouped = df.groupby("project_name").sum()

plt.figure(figsize=(10, 6))

# Create the main bar plot for 'emissions'
ax1 = sns.barplot(
    x=grouped.index, y=grouped["emissions"], color="gray", label="Emissions (kgCO2e"
)
ax1.set_ylabel("Emissions", fontsize=14)
ax1.set_xlabel("Dataset", fontsize=14)
ax1.set_title("Emissions and Energy by Dataset", fontsize=16)
ax1.legend(loc="upper left")

# Create the secondary y-axis for 'energy'
ax2 = ax1.twinx()
sns.lineplot(
    x=grouped.index,
    y=grouped["energy_consumed"],
    color="blue",
    marker="o",
    label="Energy (kWh)",
    ax=ax2,
)
ax2.set_ylabel("Energy", fontsize=14)
ax2.legend(loc="upper right")

# Adjust the x-axis labels
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig("emissions_and_energy_by_project.png")
plt.show()
