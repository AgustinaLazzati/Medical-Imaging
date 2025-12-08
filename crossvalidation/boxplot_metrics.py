import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Load the data files
files = [
    'patch_level_predictions_autoencoder_hsv_red.csv',
    'patch_level_predictions_autoencoder_mae_red.csv',
    'patch_level_predictions_autoencoder_mse.csv'
]
names = ['HSV Red', 'MAE Red', 'MSE']

custom_palette = {
    'Benign': 'blue',   # Blue
    'Malignant': 'darkturquoise' # Dark turquoise
}

plt.figure(figsize=(15, 6))

for i, (file, name) in enumerate(zip(files, names), 1):
    df = pd.read_csv(file)
    
    # Map class labels for clarity
    df['Label'] = df['true'].map({0: 'Benign', 1: 'Malignant'})
    
    plt.subplot(1, 3, i)
    
    # Assigning 'x' variable ('Label') to 'hue' to keep colors and disabling the legend
    sns.boxplot(
        data=df, 
        x='Label', 
        y='error', 
        hue='Label', 
        palette= custom_palette, 
        legend=False,
        medianprops=dict(color="#FF8C00", linewidth=1.0)
    )
    
    plt.title(f'Metric: {name}')
    plt.ylabel('Reconstruction Error')
    plt.xlabel('Ground Truth')

plt.tight_layout()
plt.savefig('error_distribution_boxplots.png')
print("Updated boxplots saved to error_distribution_boxplots.png")