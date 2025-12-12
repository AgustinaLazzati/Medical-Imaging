# AUTOENCODER ------------------------------------
## CONF 3
### RED HSV CHANNEL ERROR, 10 KFOLDS. ------------
Computing GLOBAL ROC...
GLOBAL AUC=0.9654 | Best threshold=0.000473022461
Saved: /fhome/vlia01/Medical-Imaging/crossvalidation/ROC_GLOBAL_Autoencoder_conf3_hsv_red.png

Running PATIENT-STRATIFIED K-FOLD CV...

===== FOLD 1 / 10 =====
AUC(TRAIN)=0.9569 | Th=0.000473022461
Prec=0.8556 Recall=1.0000 F1=0.9222

===== FOLD 2 / 10 =====
AUC(TRAIN)=0.9661 | Th=0.000473022461
Prec=0.9933 Recall=0.9198 F1=0.9551

===== FOLD 3 / 10 =====
AUC(TRAIN)=0.9667 | Th=0.000473022461
Prec=0.8514 Recall=1.0000 F1=0.9197

===== FOLD 4 / 10 =====
AUC(TRAIN)=0.9632 | Th=0.000473022461
Prec=0.9474 Recall=1.0000 F1=0.9730

===== FOLD 5 / 10 =====
AUC(TRAIN)=0.9665 | Th=0.000473022461
Prec=0.7941 Recall=1.0000 F1=0.8852

===== FOLD 6 / 10 =====
AUC(TRAIN)=0.9620 | Th=0.000534057617
Prec=0.9741 Recall=0.9427 F1=0.9581

===== FOLD 7 / 10 =====
AUC(TRAIN)=0.9669 | Th=0.000595092773
Prec=0.9852 Recall=0.9009 F1=0.9412

===== FOLD 8 / 10 =====
AUC(TRAIN)=0.9655 | Th=0.000473022461
Prec=0.9298 Recall=0.9815 F1=0.9550

===== FOLD 9 / 10 =====
AUC(TRAIN)=0.9748 | Th=0.000473022461
Prec=0.8427 Recall=0.9804 F1=0.9063

===== FOLD 10 / 10 =====
AUC(TRAIN)=0.9652 | Th=0.000473022461
Prec=0.9726 Recall=0.8765 F1=0.9221

===============================
 K-FOLD CROSS-VALIDATION SUMMARY
===============================
Precision : 0.9146 ± 0.0682
Recall    : 0.9602 ± 0.0443
F1-score  : 0.9338 ± 0.0258
AUC (test): 0.9654 ± 0.0043
Optimal Threshold: 0.000491333008 ± 0.000039081569

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

## CONF 2
Computing GLOBAL ROC...
GLOBAL AUC=0.9671 | Best threshold=0.000579833984
Saved: /fhome/vlia01/Medical-Imaging/crossvalidation/ROC_GLOBAL_Autoencoder_conf2_hsv_red.png

Running PATIENT-STRATIFIED K-FOLD CV...

===== FOLD 1 / 10 =====
AUC(TRAIN)=0.9585 | Th=0.000473022461
Prec=0.8533 Recall=1.0000 F1=0.9209

===== FOLD 2 / 10 =====
AUC(TRAIN)=0.9679 | Th=0.000579833984
Prec=0.9932 Recall=0.9074 F1=0.9484

===== FOLD 3 / 10 =====
AUC(TRAIN)=0.9682 | Th=0.000579833984
Prec=0.8514 Recall=1.0000 F1=0.9197

===== FOLD 4 / 10 =====
AUC(TRAIN)=0.9650 | Th=0.000579833984
Prec=0.9600 Recall=1.0000 F1=0.9796

===== FOLD 5 / 10 =====
AUC(TRAIN)=0.9681 | Th=0.000579833984
Prec=0.7941 Recall=1.0000 F1=0.8852

===== FOLD 6 / 10 =====
AUC(TRAIN)=0.9640 | Th=0.000640869141
Prec=0.9776 Recall=0.9391 F1=0.9580

===== FOLD 7 / 10 =====
AUC(TRAIN)=0.9685 | Th=0.000640869141
Prec=0.9852 Recall=0.9009 F1=0.9412

===== FOLD 8 / 10 =====
AUC(TRAIN)=0.9671 | Th=0.000579833984
Prec=0.9273 Recall=0.9444 F1=0.9358

===== FOLD 9 / 10 =====
AUC(TRAIN)=0.9763 | Th=0.000564575195
Prec=0.8427 Recall=0.9804 F1=0.9063

===== FOLD 10 / 10 =====
AUC(TRAIN)=0.9669 | Th=0.000579833984
Prec=0.9726 Recall=0.8765 F1=0.9221

===============================
 K-FOLD CROSS-VALIDATION SUMMARY
===============================
Precision : 0.9157 ± 0.0694
Recall    : 0.9549 ± 0.0452
F1-score  : 0.9317 ± 0.0255
AUC (test): 0.9671 ± 0.0042
Optimal Threshold: 0.000579833984 ± 0.000043694522

## CONF 1
Computing GLOBAL ROC...
GLOBAL AUC=0.9670 | Best threshold=0.000579833984
Saved: /fhome/vlia01/Medical-Imaging/crossvalidation/ROC_GLOBAL_Autoencoder_conf1_hsv_red.png

Running PATIENT-STRATIFIED K-FOLD CV...

===== FOLD 1 / 10 =====
AUC(TRAIN)=0.9584 | Th=0.000473022461
Prec=0.8533 Recall=1.0000 F1=0.9209

===== FOLD 2 / 10 =====
AUC(TRAIN)=0.9678 | Th=0.000579833984
Prec=0.9932 Recall=0.9012 F1=0.9450

===== FOLD 3 / 10 =====
AUC(TRAIN)=0.9681 | Th=0.000579833984
Prec=0.8514 Recall=1.0000 F1=0.9197

===== FOLD 4 / 10 =====
AUC(TRAIN)=0.9649 | Th=0.000579833984
Prec=0.9600 Recall=1.0000 F1=0.9796

===== FOLD 5 / 10 =====
AUC(TRAIN)=0.9681 | Th=0.000579833984
Prec=0.7941 Recall=1.0000 F1=0.8852

===== FOLD 6 / 10 =====
AUC(TRAIN)=0.9639 | Th=0.000732421875
Prec=0.9770 Recall=0.9140 F1=0.9444

===== FOLD 7 / 10 =====
AUC(TRAIN)=0.9685 | Th=0.000625610352
Prec=0.9852 Recall=0.9009 F1=0.9412

===== FOLD 8 / 10 =====
AUC(TRAIN)=0.9671 | Th=0.000579833984
Prec=0.9273 Recall=0.9444 F1=0.9358

===== FOLD 9 / 10 =====
AUC(TRAIN)=0.9762 | Th=0.000564575195
Prec=0.8427 Recall=0.9804 F1=0.9063

===== FOLD 10 / 10 =====
AUC(TRAIN)=0.9668 | Th=0.000579833984
Prec=0.9726 Recall=0.8765 F1=0.9221

===============================
 K-FOLD CROSS-VALIDATION SUMMARY
===============================
Precision : 0.9157 ± 0.0693
Recall    : 0.9517 ± 0.0473
F1-score  : 0.9300 ± 0.0242
AUC (test): 0.9670 ± 0.0042
Optimal Threshold: 0.000587463379 ± 0.000060363893

# VARIATIONAL AUTOENCODER
## CONF 3
Computing GLOBAL ROC...
GLOBAL AUC=0.9672 | Best threshold=0.000579833984
Saved: /fhome/vlia01/Medical-Imaging/crossvalidation/ROC_GLOBAL_Variational Autoencoder_conf3_hsv_red.png

Running PATIENT-STRATIFIED K-FOLD CV...

===== FOLD 1 / 10 =====
AUC(TRAIN)=0.9586 | Th=0.000473022461
Prec=0.8533 Recall=1.0000 F1=0.9209

===== FOLD 2 / 10 =====
AUC(TRAIN)=0.9680 | Th=0.000579833984
Prec=0.9932 Recall=0.9074 F1=0.9484

===== FOLD 3 / 10 =====
AUC(TRAIN)=0.9683 | Th=0.000579833984
Prec=0.8514 Recall=1.0000 F1=0.9197

===== FOLD 4 / 10 =====
AUC(TRAIN)=0.9651 | Th=0.000579833984
Prec=0.9600 Recall=1.0000 F1=0.9796

===== FOLD 5 / 10 =====
AUC(TRAIN)=0.9682 | Th=0.000579833984
Prec=0.7941 Recall=1.0000 F1=0.8852

===== FOLD 6 / 10 =====
AUC(TRAIN)=0.9641 | Th=0.000640869141
Prec=0.9776 Recall=0.9391 F1=0.9580

===== FOLD 7 / 10 =====
AUC(TRAIN)=0.9686 | Th=0.000640869141
Prec=0.9852 Recall=0.9009 F1=0.9412

===== FOLD 8 / 10 =====
AUC(TRAIN)=0.9672 | Th=0.000579833984
Prec=0.9273 Recall=0.9444 F1=0.9358

===== FOLD 9 / 10 =====
AUC(TRAIN)=0.9764 | Th=0.000564575195
Prec=0.8427 Recall=0.9804 F1=0.9063

===== FOLD 10 / 10 =====
AUC(TRAIN)=0.9670 | Th=0.000579833984
Prec=0.9726 Recall=0.8765 F1=0.9221

===============================
 K-FOLD CROSS-VALIDATION SUMMARY
===============================
Precision : 0.9157 ± 0.0694
Recall    : 0.9549 ± 0.0452
F1-score  : 0.9317 ± 0.0255
AUC (test): 0.9672 ± 0.0042
Optimal Threshold: 0.000579833984 ± 0.000043694522

## CONF 2
Computing GLOBAL ROC...
GLOBAL AUC=0.9672 | Best threshold=0.000579833984
Saved: /fhome/vlia01/Medical-Imaging/crossvalidation/ROC_GLOBAL_Variational Autoencoder_conf2_hsv_red.png

Running PATIENT-STRATIFIED K-FOLD CV...

===== FOLD 1 / 10 =====
AUC(TRAIN)=0.9586 | Th=0.000473022461
Prec=0.8533 Recall=1.0000 F1=0.9209

===== FOLD 2 / 10 =====
AUC(TRAIN)=0.9680 | Th=0.000579833984
Prec=0.9932 Recall=0.9074 F1=0.9484

===== FOLD 3 / 10 =====
AUC(TRAIN)=0.9683 | Th=0.000579833984
Prec=0.8514 Recall=1.0000 F1=0.9197

===== FOLD 4 / 10 =====
AUC(TRAIN)=0.9651 | Th=0.000579833984
Prec=0.9600 Recall=1.0000 F1=0.9796

===== FOLD 5 / 10 =====
AUC(TRAIN)=0.9682 | Th=0.000579833984
Prec=0.7941 Recall=1.0000 F1=0.8852

===== FOLD 6 / 10 =====
AUC(TRAIN)=0.9641 | Th=0.000640869141
Prec=0.9776 Recall=0.9391 F1=0.9580

===== FOLD 7 / 10 =====
AUC(TRAIN)=0.9686 | Th=0.000640869141
Prec=0.9852 Recall=0.9009 F1=0.9412

===== FOLD 8 / 10 =====
AUC(TRAIN)=0.9672 | Th=0.000579833984
Prec=0.9273 Recall=0.9444 F1=0.9358

===== FOLD 9 / 10 =====
AUC(TRAIN)=0.9764 | Th=0.000564575195
Prec=0.8427 Recall=0.9804 F1=0.9063

===== FOLD 10 / 10 =====
AUC(TRAIN)=0.9670 | Th=0.000579833984
Prec=0.9726 Recall=0.8765 F1=0.9221

===============================
 K-FOLD CROSS-VALIDATION SUMMARY
===============================
Precision : 0.9157 ± 0.0694
Recall    : 0.9549 ± 0.0452
F1-score  : 0.9317 ± 0.0255
AUC (test): 0.9672 ± 0.0042
Optimal Threshold: 0.000579833984 ± 0.000043694522


