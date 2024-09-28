import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV with the results (replace with your file path)
df = pd.read_csv("results.csv")

# Plot the AUROC metrics for each dataset
plt.figure(figsize=(10, 6))

# Plot Instance AUROC
plt.plot( df['instance_auroc'], label='Instance AUROC', marker='o', color='blue')

# Plot Full Pixel AUROC
plt.plot( df['full_pixel_auroc'], label='Full Pixel AUROC', marker='o', color='green')

# Plot Anomaly Pixel AUROC
plt.plot(df['anomaly_pixel_auroc'], label='Anomaly Pixel AUROC', marker='o', color='red')

# Add title and labels
plt.title('AUROC Scores for Different Datasets')
plt.xlabel('Dataset Name')
plt.ylabel('AUROC Score')

# Add legend
plt.legend()

# Rotate x-axis labels if necessary
plt.xticks(rotation=45)

# Add grid for readability
plt.grid(True)

# Save the figure as a PNG file (optional)
plt.savefig('auroc_scores.png')

# Show the plot
plt.tight_layout()  # Adjust layout to make space for labels
plt.show()
