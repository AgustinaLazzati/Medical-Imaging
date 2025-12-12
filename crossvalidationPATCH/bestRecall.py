import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# List of all files
files = [
    'patch_level_predictions_Autoencoder_conf1_hsv_red.csv',
    'patch_level_predictions_Autoencoder_conf2_hsv_red.csv',
    'patch_level_predictions_Autoencoder_conf3_hsv_red.csv',
    'patch_level_predictions_Variational_Autoencoder_conf2_hsv_red.csv',
    'patch_level_predictions_Variational_Autoencoder_conf3_hsv_red.csv'
]

# Human readable names
names = ['AE Conf 1', 'AE Conf 2', 'AE Conf 3',
         'VAE Conf 2', 'VAE Conf 3']

results = []

# ---- Compute per-model recall ----
for file, name in zip(files, names):

    df = pd.read_csv(file)

    TP = ((df.true == 1) & (df.pred == 1)).sum()
    FN = ((df.true == 1) & (df.pred == 0)).sum()
    TN = ((df.true == 0) & (df.pred == 0)).sum()
    FP = ((df.true == 0) & (df.pred == 1)).sum()

    recall_pos = TP / (TP + FN) if (TP + FN) > 0 else 0
    recall_neg = TN / (TN + FP) if (TN + FP) > 0 else 0

    results.append({
        'Model': name,
        'Recall+ (Positive)': recall_pos,
        'Recall- (Negative)': recall_neg
    })

# Convert to DataFrame
metrics_df = pd.DataFrame(results)

# Melt long-form for seaborn
metrics_melted = metrics_df.melt(
    id_vars='Model',
    value_vars=['Recall+ (Positive)', 'Recall- (Negative)'],
    var_name='Metric',
    value_name='Value'
)

# ---- Plot barplot ----
plt.figure(figsize=(14, 6))

ax = sns.barplot(
    data=metrics_melted,
    x='Model',
    y='Value',
    hue='Metric',
    palette=['#FF8C00', '#1E90FF']  # orange = recall+, blue = recall-
)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3)
plt.ylim(0, 1)
plt.ylabel('Recall')
plt.title('Recall+ (Positive) and Recall- (Negative) Across AE/VAE Configurations', fontsize=16)
plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig('recallVAEvsAE_allconf.png')

print("Saved recallVAEvsAE_allconf.png")
