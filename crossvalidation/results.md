## AUTOENCODER RED HSV CHANNEL ERROR, 10 KFOLDS. 
Total number of samples: 2695
Total samples in dataset: 2695
Benign patches: 1216
Malignant patches: 1475

Computing GLOBAL ROC...
GLOBAL AUC=0.9654 | Best threshold=0.000473
Saved: /fhome/vlia01/Medical-Imaging/crossvalidation/ROC_GLOBAL_Autoen

Running PATIENT-STRATIFIED K-FOLD CV...

===== FOLD 1 / 10 =====
AUC(TRAIN)=0.9569 | Th=0.000473
Prec=0.8556 Recall=1.0000 F1=0.9222

===== FOLD 2 / 10 =====
AUC(TRAIN)=0.9661 | Th=0.000473
Prec=0.9933 Recall=0.9198 F1=0.9551

===== FOLD 3 / 10 =====
AUC(TRAIN)=0.9667 | Th=0.000473
Prec=0.8514 Recall=1.0000 F1=0.9197

===== FOLD 4 / 10 =====
AUC(TRAIN)=0.9632 | Th=0.000473
Prec=0.9474 Recall=1.0000 F1=0.9730

===== FOLD 5 / 10 =====
AUC(TRAIN)=0.9665 | Th=0.000473
Prec=0.7941 Recall=1.0000 F1=0.8852

===== FOLD 6 / 10 =====
AUC(TRAIN)=0.9620 | Th=0.000534
Prec=0.9741 Recall=0.9427 F1=0.9581

===== FOLD 7 / 10 =====
AUC(TRAIN)=0.9669 | Th=0.000595
Prec=0.9852 Recall=0.9009 F1=0.9412

===== FOLD 8 / 10 =====
AUC(TRAIN)=0.9655 | Th=0.000473
Prec=0.9298 Recall=0.9815 F1=0.9550

===== FOLD 9 / 10 =====
AUC(TRAIN)=0.9748 | Th=0.000473
Prec=0.8427 Recall=0.9804 F1=0.9063

===== FOLD 10 / 10 =====
AUC(TRAIN)=0.9652 | Th=0.000473
Prec=0.9726 Recall=0.8765 F1=0.9221

===============================
 K-FOLD CROSS-VALIDATION SUMMARY
===============================
Precision : 0.9146 ± 0.0682
Recall    : 0.9602 ± 0.0443
F1-score  : 0.9338 ± 0.0258
AUC (test): 0.9654 ± 0.0043

