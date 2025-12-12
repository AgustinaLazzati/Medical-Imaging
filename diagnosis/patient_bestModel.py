import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load both models
df_vae = pd.read_csv('patient_predictions_diagnosisVAE(meanTHR).csv')
df_ae  = pd.read_csv('patient_predictions_diagnosisAE(globalTHR).csv')
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Load the data files
files = [
    'patient_predictions_diagnosisVAE(meanTHR).csv',
    'patient_predictions_diagnosisAE(globalTHR).csv'
]
names = ['Variational Autoencoder conf3', 'Autoencoder conf3']

custom_palette = {
    'NEGATIVE': 'white',   # Blue
    'POSITIVE': 'white' # Dark turquoise
}

plt.figure(figsize=(15, 6))

for i, (file, name) in enumerate(zip(files, names), 1):
    df = pd.read_csv(file)
    
    # Map class labels for clarity
    df['Label'] = df['label'].map({0: 'NEGATIVE', 1: 'POSITIVE'})
    
    plt.subplot(1, 2, i)
    
    # Assigning 'x' variable ('Label') to 'hue' to keep colors and disabling the legend
    sns.boxplot(
        data=df, 
        x='Label', 
        y='score', 
        hue='Label', 
        palette= custom_palette, 
        legend=False,
        medianprops=dict(color="#FF8C00", linewidth=2),
        boxprops=dict(edgecolor='black', linewidth=1.5),
        whiskerprops=dict(color='black', linewidth=1.5),
        capprops=dict(color='black', linewidth=1.5),
    )
    
    plt.title(f'Diagnosis Scores: {name}')
    plt.ylabel('Patient-level Anomaly Score')
    plt.xlabel('Ground Truth')

plt.tight_layout()
plt.savefig('AE_vs_VAE_boxplots.png')
print("Updated boxplots saved to AE_vs_VAE_boxplots.png")



results = []

# ---- Compute per-model recall ----
for file, name in zip(files, names):

    df = pd.read_csv(file)

    TP = ((df.label == 1) & (df.prediction == 1)).sum()
    FN = ((df.label == 1) & (df.prediction == 0)).sum()
    TN = ((df.label == 0) & (df.prediction == 0)).sum()
    FP = ((df.label == 0) & (df.prediction == 1)).sum()

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
plt.title('PATIENT LEVEL: Recall+ (Positive) and Recall- (Negative) VAE VS AE', fontsize=16)
plt.xticks(rotation=25)
plt.tight_layout()

plt.savefig('patientrecallVAEvsAE_allconf.png')

print("Saved patientrecallVAEvsAE_allconf.png")
