import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Load the data files
files = [
    'patch_level_predictions_Autoencoder_conf1_hsv_red.csv',
    'patch_level_predictions_Autoencoder_conf2_hsv_red.csv',
    'patch_level_predictions_Autoencoder_conf3_hsv_red.csv'
]
names = ['Config 1', 'Config 2', 'Config 3']

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
    
    plt.title(f'AUTOENCODER CONFIGURATION: {name}')
    plt.ylabel('Reconstruction Error')
    plt.xlabel('Ground Truth')

plt.tight_layout()
plt.savefig('AE_confs_boxplots.png')
print("Updated boxplots saved to AE_confs_boxplots.png")