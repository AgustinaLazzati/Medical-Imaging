# 🧬 Automatic Helicobacter pylori Diagnosis

## 📌 Project Overview

Helicobacter pylori (*H. pylori*) is a bacterium strongly associated with gastritis and gastric cancer. 
Its diagnosis from immunohistochemistry (IHC) slides is traditionally performed by pathologists through 
manual visual inspection, a process that is time-consuming and prone to subjectivity, especially in 
low-density cases.

This project explores **deep learning–based automated systems** to assist in the detection of *H. pylori*
from histopathology images. The goal is to provide a **robust patient-level diagnosis** by aggregating
patch-level predictions extracted from whole-slide images.

Two different approaches are implemented and compared:

- **System I:** Anomaly Detection using Autoencoders  
- **System II:** Patient-level Classification using a Gated Attention Mechanism  

---

## 🎯 Challenge Objective

- Detect *H. pylori* presence from IHC whole-slide images
- Perform **patch-level analysis** and aggregate results into a **patient-level diagnosis**
- Compare different deep learning architectures and design choices
- Evaluate robustness using cross-validation and a holdout test set

---

## 🧠 Methodology

The proposed pipeline consists of three main stages:

1. **Patch Extraction (Preprocessed)**  
   The dataset provides 256×256 image patches extracted from tissue borders, where *H. pylori* typically appears.

2. **Patch-level Analysis**
   - Anomaly detection via reconstruction error (System I)
   - Feature embedding and attention-based aggregation (System II)

3. **Patient-level Diagnosis**
   - Patch predictions are aggregated to produce a final binary diagnosis per patient

<img width="1310" height="532" alt="GENERAL_pipeline" src="https://github.com/user-attachments/assets/37c63924-aa33-4f42-8ccf-79bf4ff20928" />

---

## 🔍 System I: Anomaly Detection via Autoencoders

System I formulates *H. pylori* detection as an **anomaly detection problem**.

### Key Ideas
- Models are trained **only on patches from H. pylori–negative patients**
- The model learns the distribution of normal tissue
- Patches containing bacteria produce **high reconstruction error**
- Patch predictions are aggregated to diagnose each patient

### Models
- Convolutional Autoencoder (AE)
- Variational Autoencoder (VAE)

### Reconstruction Error Metrics
Three reconstruction error definitions were evaluated:

- Mean Squared Error (MSE)
- Mean Absolute Error on the Red Channel
- **HSV-based Red Pixel Reconstruction Error (selected)**

![reconstruction (1)](https://github.com/user-attachments/assets/10acd761-3d27-4fdf-9acf-54467862e6d9)

The HSV-based metric explicitly captures biologically relevant red staining and achieved the best
performance across all experiments.

---

## 👁️ System II: Attention-Based Patient Classification

System II treats each patient as a **bag of image patches** and learns a single patient representation.

### Pipeline
1. Extract patch embeddings using the encoder from System I
2. Apply a **Gated Attention Mechanism** to weight informative patches
3. Aggregate patches into a patient-level representation
4. Perform binary classification

This approach allows the model to focus on the most relevant regions while suppressing background noise.

---

## 📊 Dataset

This project uses the **Quiron dataset**, a collection of 245 whole-slide immunohistochemistry images of
gastric tissue, each corresponding to a different patient.

### Data Distribution
It consists of **245 patients**, with each patient contributing a single whole-slide image (WSI). For the purposes of binary classification, the diagnostic labels are organized as follows:

* **Negative Group (117 patients):** Labeled as `NEGATIVA`.
* **Positive Group (128 patients):** Consolidates the `BAIXA` (Low) and `ALTA` (High) categories.

---

### Image Preparation and Validation
To optimize the data for computational analysis, the following preprocessing steps were applied:

1.  **Patch Generation:** WSIs were segmented into **$256 \times 256$ pixel** tiles, focusing primarily on the tissue's marginal areas.
2.  **Expert Annotation:** A specific portion of the dataset features patches that were manually labeled by specialists.
3.  **Experimental Setup:** This annotated subset is reserved for critical pipeline stages, including:
    * Cross-validation
    * Model validation
    * Selection of classification thresholds


## 📚 References
Cano, P., Caravaca, Á., Gil, D., & Musulen, E.  
   *Diagnosis of Helicobacter pylori using autoencoders for the detection of anomalous staining patterns
   in immunohistochemistry images*.  
   arXiv preprint, 2023.  

