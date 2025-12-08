# AUTOENCODER ------------------------------------
### RED HSV CHANNEL ERROR, 10 KFOLDS. ------------
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

### MSE CHANNEL ERROR, 10 KFOLDS. ------------

Computing GLOBAL ROC...
GLOBAL AUC=0.7288 | Best threshold=0.000109542299
Saved: /fhome/vlia01/Medical-Imaging/crossvalidation/ROC_GLOBAL_Autoencoder_mse.png

Running PATIENT-STRATIFIED K-FOLD CV...

===== FOLD 1 / 10 =====
AUC(TRAIN)=0.7453 | Th=0.000105880892
Prec=0.5147 Recall=0.9875 F1=0.6767

===== FOLD 2 / 10 =====
AUC(TRAIN)=0.7174 | Th=0.000109556677
Prec=0.8521 Recall=0.8889 F1=0.8701

===== FOLD 3 / 10 =====
AUC(TRAIN)=0.7304 | Th=0.000220583752
Prec=0.5238 Recall=0.3492 F1=0.4190

===== FOLD 4 / 10 =====
AUC(TRAIN)=0.7241 | Th=0.000109542299
Prec=0.6452 Recall=0.8333 F1=0.7273

===== FOLD 5 / 10 =====
AUC(TRAIN)=0.7269 | Th=0.000109542299
Prec=0.6265 Recall=0.9630 F1=0.7591

===== FOLD 6 / 10 =====
AUC(TRAIN)=0.7097 | Th=0.000165292367
Prec=0.8197 Recall=0.6846 F1=0.7461

===== FOLD 7 / 10 =====
AUC(TRAIN)=0.7577 | Th=0.000165292367
Prec=0.8462 Recall=0.1486 F1=0.2529

===== FOLD 8 / 10 =====
AUC(TRAIN)=0.7319 | Th=0.000177942144
Prec=0.6000 Recall=0.1667 F1=0.2609

===== FOLD 9 / 10 =====
AUC(TRAIN)=0.7276 | Th=0.000109542299
Prec=0.6972 Recall=0.9935 F1=0.8194

===== FOLD 10 / 10 =====
AUC(TRAIN)=0.7372 | Th=0.000177942144
Prec=0.6190 Recall=0.1605 F1=0.2549

===============================
 K-FOLD CROSS-VALIDATION SUMMARY
===============================
Precision : 0.6744 ± 0.1194
Recall    : 0.6176 ± 0.3502
F1-score  : 0.5786 ± 0.2393
AUC (test): 0.7308 ± 0.0129
Optimal Threshold: 0.000145111724 ± 0.000039070434

### MAE RED CHANNEL ERROR, 10 KFOLDS. ------------
GLOBAL AUC=0.6560 | Best threshold=0.005743546411
Saved: /fhome/vlia01/Medical-Imaging/crossvalidation/ROC_GLOBAL_Autoencoder_mae_red.png

Running PATIENT-STRATIFIED K-FOLD CV...

===== FOLD 1 / 10 =====
AUC(TRAIN)=0.6793 | Th=0.005743545480
Prec=0.4856 Recall=1.0000 F1=0.6537

===== FOLD 2 / 10 =====
AUC(TRAIN)=0.6426 | Th=0.005743546411
Prec=0.8192 Recall=0.8951 F1=0.8555

===== FOLD 3 / 10 =====
AUC(TRAIN)=0.6590 | Th=0.009433444589
Prec=0.4211 Recall=0.2540 F1=0.3168

===== FOLD 4 / 10 =====
AUC(TRAIN)=0.6510 | Th=0.005743546411
Prec=0.5868 Recall=0.9861 F1=0.7358

===== FOLD 5 / 10 =====
AUC(TRAIN)=0.6547 | Th=0.005743545480
Prec=0.5349 Recall=0.8519 F1=0.6571

===== FOLD 6 / 10 =====
AUC(TRAIN)=0.6248 | Th=0.005743545480
Prec=0.6997 Recall=0.8853 F1=0.7816

===== FOLD 7 / 10 =====
AUC(TRAIN)=0.6818 | Th=0.009304432198
Prec=0.5294 Recall=0.0405 F1=0.0753

===== FOLD 8 / 10 =====
AUC(TRAIN)=0.6582 | Th=0.009304432198
Prec=0.0000 Recall=0.0000 F1=0.0000

===== FOLD 9 / 10 =====
AUC(TRAIN)=0.6606 | Th=0.005743545946
Prec=0.6426 Recall=0.9869 F1=0.7784

===== FOLD 10 / 10 =====
AUC(TRAIN)=0.6650 | Th=0.006034692749
Prec=0.5488 Recall=0.5556 F1=0.5521

===============================
 K-FOLD CROSS-VALIDATION SUMMARY
===============================
Precision : 0.5268 ± 0.2058
Recall    : 0.6455 ± 0.3827
F1-score  : 0.5406 ± 0.2893
AUC (test): 0.6577 ± 0.0157
Optimal Threshold: 0.006853827694 ± 0.001635013292