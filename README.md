# Optimizing Conformal Prediction Sets for Pathological Image Classification




## Abstract

The intersection of Deep Learning (DL) and pathology
has gained significant attention, encompassing cell clas-
sification, detection, segmentation, and whole-slide image
(WSI) analysis. Further works at this intersection have in-
creasingly focused on integrating uncertainty quantification
(UQ) with DL methods for pathology to address their occa-
sional unreliability in clinical settings. Conformal Predic-
tion (CP) is one of the UQ methods deployed for various
medical settings including pathology. CP methods are com-
putationally efficient and offer user-defined coverage guar-
antees to generate prediction sets that include the true label.
However, CP methods lack inherent control over the com-
positionality of prediction sets, which restricts their clinical
utility. This study presents a novel hinge loss-based training
method for the underlying models used in CP methods. This
approach aims to provide effective control over the compo-
sitionality of prediction sets, aligning more closely with the
specific needs of pathologists. We evaluate the effectiveness
of this training approach using three application-specific
metrics tailored to enhance the integration of CP methods
into clinical pathology workflows. Our results show that the
Hinge Loss-based training approach outperforms the tradi-
tional Cross-Entropy method across all evaluation metrics,
effectively managing the compositionality of conformal pre-
diction sets.


## Dataset Guide
### Cervical Cancer dataset:-
**Center for Recognition and Inspection of Cells (CRIC) Dataset** : [Download Link](https://database.cric.com.br/downloads)

### Breast Cancer dataset:-
**Breast Cancer Histopathological Database (BreakHis)**: [Download Link](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)




### Extract Cells from the Patches 
- To extract a cell crop size of 100 x 100 centered on the nucleus from the patches.
```
python Cropping.py --dataset='path to the cell centre coordinates csv file' --img_dir='path to the cric img patches directory' --cell_img_dir='path to cell img directory'
```

### Feature_Extractor
- To extract features from the image patches.
```
python Feature_Extractor.py --img_dir='path to the image directory' --csv_file='path to the csv file' --model='Model to be used for feature extraction' --batch_size='Batch size for the dataloader' --num_workers='num workers for dataloader'
```

### Train-Val-Test Split for image features 
- To split the image feature containing csv file into train, val and test.
```
python feature_diet_Train_val_test_split.py --dataset='path to the img feature file' --split='Train/Test split ratio' --folds='No of folds in K-folds'
```

### Train the model in order to get softmax output for test data points.

- Train the model using cross entropy loss
```
python feature_main.py --loss='cross_entropy' --feat_dir='path to the feature directory' --num_epochs='Number of total training epochs' --model='Model to be used' --batch_size='Batch size for the dataloader' --num_workers='num workers for dataloader' --dataset='Dataset used :- Breast_cancer/Cervical_cancer '
```

- Train the model using hinge loss for Risk-Based Controlled Set Sizes i.e :- C(High Grade) > C(Low Grade) > C(Normal) 

```
python feature_main.py --loss='hinge_loss' --metric='expt2' --distance_in_hinge_loss='distance_in_hinge_loss'--feat_dir='path to the feature directory' --num_epochs='Number of total training epochs' --model='Model to be used' --batch_size='Batch size for the dataloader' --num_workers='num workers for dataloader' --dataset='Dataset used :- Breast_cancer/Cervical_cancer '
```

- Train the model using hinge loss for reducing mix of various risks
```
python feature_main.py --loss='hinge_loss' --metric='class_Overlap_metric' --distance_in_hinge_loss='distance_in_hinge_loss'--feat_dir='path to the feature directory' --num_epochs='Number of total training epochs' --model='Model to be used' --batch_size='Batch size for the dataloader' --num_workers='num workers for dataloader' --dataset='Dataset used :- Breast_cancer/Cervical_cancer '
```

- Train the model using hinge loss for avoiding confusing classes
```
python feature_main.py --loss='hinge_loss' --metric='confusion_set_Overlap_metric' --distance_in_hinge_loss='distance_in_hinge_loss'--feat_dir='path to the feature directory' --num_epochs='Number of total training epochs' --model='Model to be used' --batch_size='Batch size for the dataloader' --num_workers='num workers for dataloader' --dataset='Dataset used :- Breast_cancer/Cervical_cancer '
```

### Evaluate Conformal Method

- Eval CP method for baseline results on cervical cancer dataset
```
python Cervical_cancer_Eval_CP_method --Trials='Number of total trials for eval CP method' --softmax_output_file_path='path to the softmax_output_file' --expt_no='1' --split='Calib/test split ratio' --CP_method='CP method to be used' --alpha='value of alpha for CP coverage' 
```

- Eval CP method for Risk-Based Controlled Set Sizes on cervical cancer dataset

```
python Cervical_cancer_Eval_CP_method --Trials='Number of total trials for eval CP method' --softmax_output_file_path='path to the softmax_output_file' --expt_no='2' --split='Calib/test split ratio' --CP_method='CP method to be used' --alpha='value of alpha for CP coverage' 
```

- Eval CP method for reducing mix of various risks on cervical cancer dataset

```
python Cervical_cancer_Eval_CP_method --Trials='Number of total trials for eval CP method' --softmax_output_file_path='path to the softmax_output_file' --expt_no='3' --split='Calib/test split ratio' --CP_method='CP method to be used' --alpha='value of alpha for CP coverage' 
```

- Eval CP method for avoiding confusing classes on cervical cancer dataset
```
python Cervical_cancer_Eval_CP_method --Trials='Number of total trials for eval CP method' --softmax_output_file_path='path to the softmax_output_file' --expt_no='4' --split='Calib/test split ratio' --CP_method='CP method to be used' --alpha='value of alpha for CP coverage' 
```

- Eval CP method for baseline results on Breast_cancer dataset
```
python Breast_Cancer_Eval_CP_method --Trials='Number of total trials for eval CP method' --softmax_output_file_path='path to the softmax_output_file' --expt_no='1' --split='Calib/test split ratio' --CP_method='CP method to be used' --alpha='value of alpha for CP coverage' 
```

- Eval CP method for Risk-Based Controlled Set Sizes on Breast_cancer dataset
```
python Breast_Cancer_Eval_CP_method --Trials='Number of total trials for eval CP method' --softmax_output_file_path='path to the softmax_output_file' --expt_no='2' --split='Calib/test split ratio' --CP_method='CP method to be used' --alpha='value of alpha for CP coverage' 
```

- Eval CP method for reducing mix of various risks on Breast_cancer dataset
```
python Breast_Cancer_Eval_CP_method --Trials='Number of total trials for eval CP method' --softmax_output_file_path='path to the softmax_output_file' --expt_no='3' --split='Calib/test split ratio' --CP_method='CP method to be used' --alpha='value of alpha for CP coverage' 
```

- Eval CP method for avoiding confusing classes on cervical cancer dataset
```
python Breast_Cancer_Eval_CP_method --Trials='Number of total trials for eval CP method' --softmax_output_file_path='path to the softmax_output_file' --expt_no='4' --split='Calib/test split ratio' --CP_method='CP method to be used' --alpha='value of alpha for CP coverage' 
```


###  Grid search for hyperparameter tuning(k_reg, lambd) for RAPS method on breast cancer dataset
- see
```
breast-cancer-raps-grid-search.ipynb
```

###  Grid search for hyperparameter tuning(k_reg, lambd) for RAPS method on cervical cancer dataset
- see
```
Cervical_cancer_raps_grid_search.ipynb
```








