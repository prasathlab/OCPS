import torch.nn as nn
import torch 
import torch.nn.functional as F
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score








def Multiclass_classification_metrices(y_true, y_pred, num_classes):

    accuracy = accuracy_score(y_true, y_pred)
    
    # AUC :- 
    auc = []
    for i in range(num_classes):
        y_true_class = [1 if label == i else 0 for label in y_true]
        y_pred_class = [1 if pred == i else 0 for pred in y_pred]
        try:
            auc_class = roc_auc_score(y_true_class, y_pred_class)
        except ValueError as e:
            print("ValueError occurred:", e)
            auc_class = 0
        auc.append(auc_class)
    macro_AUC = sum(auc)/num_classes

    return macro_AUC, accuracy





def avg_set_size_metric(conformal_set):
    lengths = torch.sum(conformal_set, dim=1)
    avg_set_size_len = torch.sum(lengths)/conformal_set.shape[0]
    return avg_set_size_len



def coverage_gap_metric(conformal_set, df_true_class_test, alpha):
    true_class = conformal_set[range(conformal_set.shape[0]), df_true_class_test]
    tensor_sum = torch.sum(true_class)
    coverage = tensor_sum/true_class.shape[0]
    coverage_gap = (abs((1-alpha) - coverage)/(1-alpha))*100
    return coverage_gap, coverage




def hinge_loss(metric, output_logits, label, distance_in_hinge_loss, dataset):
    loss = 0
    
    x = distance_in_hinge_loss
    
    if dataset == 'Cervical_cancer' :

        for i in range(output_logits.shape[0]):
            true_class_logits = output_logits[i][label[i]]
            current_logits = output_logits[i]

            if metric == 'class_Overlap_metric':
                if label[i] == 0 or label[i] == 1 or label[i] == 2:
                    #margin_list = [1, 1, 1, 5, 5, 5, 5, 5]
                    margin_list = [1, 1, 1, x, x, x, x, x]
                    margin_list[label[i]] = 0
                    
                elif label[i] == 4 or label[i] == 6 or label[i] == 7:
                    #margin_list = [5, 5, 5, 5, 1, 5, 1, 1]
                    margin_list = [x, x, x, x, 1, x, 1, 1]
                    margin_list[label[i]] = 0

                elif label[i] == 3 or label[i] == 5:
                    #margin_list = [5, 5, 5, 1, 5, 1, 5, 5]
                    margin_list = [x, x, x, 1, x, 1, x, x]
                    margin_list[label[i]] = 0



            elif metric == 'expt2':
                if label[i] == 0 or label[i] == 1 or label[i] == 2:
                    #margin_list = [5, 5, 5, 5, 5, 5, 5, 5]
                    #margin_list = [20, 20, 20, 20, 20, 20, 20, 20]
                    margin_list = [x, x, x, x, x, x, x, x]
                    margin_list[label[i]] = 0
                    
                elif label[i] == 4 or label[i] == 6 or label[i] == 7:
                    #margin_list = [5, 5, 5, 5, 1, 5, 1, 1]
                    #margin_list = [10, 10, 10, 10, 1, 10, 1, 1]
                    margin_list = [x, x, x, 1, 1, 1, 1, 1]
                    margin_list[label[i]] = 0

                elif label[i] == 3 or label[i] == 5:
                    #margin_list = [5, 5, 5, 1, 5, 1, 5, 5]
                    #margin_list = [15, 15, 15, 1, 15, 1, 15, 15]
                    margin_list = [x, x, x, 1, x, 1, x, x]
                    margin_list[label[i]] = 0




            elif metric == 'confusion_set_Overlap_metric':
                if label[i] == 4:
                    #margin_list = [1, 1, 1, 1, 1, 1, 5, 1]
                    margin_list = [1, 1, 1, 1, 1, 1, x, 1]
                    margin_list[label[i]] = 0
                    
                elif label[i] == 6:
                    #margin_list = [1, 1, 1, 1, 5, 1, 1, 1]
                    margin_list = [1, 1, 1, 1, x, 1, 1, 1]
                    margin_list[label[i]] = 0

                elif label[i] == 3:
                    #margin_list = [1, 1, 1, 1, 1, 5, 1, 1]
                    margin_list = [1, 1, 1, 1, 1, x, 1, 1]
                    margin_list[label[i]] = 0

                elif label[i] == 5:
                    #margin_list = [1, 1, 1, 5, 1, 1, 1, 1]
                    margin_list = [1, 1, 1, x, 1, 1, 1, 1]
                    margin_list[label[i]] = 0
                
                else:
                    margin_list = [1, 1, 1, 1, 1, 1, 1, 1]
                    margin_list[label[i]] = 0


            loss_0 = max(0, (current_logits[0] - true_class_logits + margin_list[0]))
            loss_1 = max(0, (current_logits[1] - true_class_logits + margin_list[1]))
            loss_2 = max(0, (current_logits[2] - true_class_logits + margin_list[2]))
            loss_3 = max(0, (current_logits[3] - true_class_logits + margin_list[3]))
            loss_4 = max(0, (current_logits[4] - true_class_logits + margin_list[4]))
            loss_5 = max(0, (current_logits[5] - true_class_logits + margin_list[5]))
            loss_6 = max(0, (current_logits[6] - true_class_logits + margin_list[6]))
            loss_7 = max(0, (current_logits[7] - true_class_logits + margin_list[7]))

            current_loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7

            loss += current_loss

        return loss/output_logits.shape[0]


    
    
    elif dataset == 'Breast_cancer' :

        for i in range(output_logits.shape[0]):
            true_class_logits = output_logits[i][label[i]]
            current_logits = output_logits[i]


            if metric == 'expt2':
                if label[i] == 0 or label[i] == 1 or label[i] == 2 or label[i] == 3:
                    margin_list = [x, x, x, x, x, x, x, x]
                    margin_list[label[i]] = 0

                elif label[i] == 4 or label[i] == 5 or label[i] == 6 or label[i] == 7:
                    margin_list = [1, 1, 1, 1, 1, 1, 1, 1]
                    margin_list[label[i]] = 0



            elif metric == 'class_Overlap_metric':
                if label[i] == 0 or label[i] == 1 or label[i] == 2 or label[i] == 3:
                    margin_list = [1, 1, 1, 1, x, x, x, x]
                    margin_list[label[i]] = 0

                elif label[i] == 4 or label[i] == 5 or label[i] == 6 or label[i] == 7:
                    margin_list = [x, x, x, x, 1, 1, 1, 1]
                    margin_list[label[i]] = 0



            elif metric == 'confusion_set_Overlap_metric':
                if label[i] == 1:
                    #margin_list = [1, 1, 1, 1, 1, 1, 5, 1]
                    margin_list = [1, 1, x, 1, 1, 1, 1, 1]
                    margin_list[label[i]] = 0
                    
                elif label[i] == 2:
                    #margin_list = [1, 1, 1, 1, 5, 1, 1, 1]
                    margin_list = [1, x, 1, 1, 1, 1, 1, 1]
                    margin_list[label[i]] = 0

                elif label[i] == 4:
                    #margin_list = [1, 1, 1, 1, 5, 1, 1, 1]
                    margin_list = [1, 1, 1, 1, 1, x, 1, 1]
                    margin_list[label[i]] = 0

                elif label[i] == 5:
                    #margin_list = [1, 1, 1, 1, 5, 1, 1, 1]
                    margin_list = [1, 1, 1, 1, x, 1, 1, 1]
                    margin_list[label[i]] = 0

                else:
                    margin_list = [1, 1, 1, 1, 1, 1, 1, 1]
                    margin_list[label[i]] = 0




    
            loss_0 = max(0, (current_logits[0] - true_class_logits + margin_list[0]))
            loss_1 = max(0, (current_logits[1] - true_class_logits + margin_list[1]))
            loss_2 = max(0, (current_logits[2] - true_class_logits + margin_list[2]))
            loss_3 = max(0, (current_logits[3] - true_class_logits + margin_list[3]))
            loss_4 = max(0, (current_logits[4] - true_class_logits + margin_list[4]))
            loss_5 = max(0, (current_logits[5] - true_class_logits + margin_list[5]))
            loss_6 = max(0, (current_logits[6] - true_class_logits + margin_list[6]))
            loss_7 = max(0, (current_logits[7] - true_class_logits + margin_list[7]))
            current_loss = loss_0 + loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7

            loss += current_loss

        #loss = loss/output_logits.shape[0]

        #loss = torch.tensor(loss, requires_grad=True)


        return loss/output_logits.shape[0]




def breast_cancer_class_Overlap_metric(conformal_set, label):

    overlap_count = 0

    for i in range(conformal_set.shape[0]):
        current_set = conformal_set[i]

        if (label[i] == 0 or label[i] == 1 or label[i] == 2 or label[i] == 3 ) and (current_set[4] == 1 or current_set[5] == 1 or current_set[6] == 1 or current_set[7] == 1):
            overlap_count += 1
        
        elif (label[i] == 4 or label[i] == 5 or label[i] == 6 or label[i] == 7 ) and (current_set[0] == 1 or current_set[1] == 1 or current_set[2] == 1 or current_set[3] == 1):
            overlap_count += 1


    perecentage_of_overlap  = (overlap_count/conformal_set.shape[0])*100

    return perecentage_of_overlap

     




def class_Overlap_metric(conformal_set, label):

    overlap_count = 0

    for i in range(conformal_set.shape[0]):
        current_set = conformal_set[i]

        if (label[i] == 0 or label[i] == 1 or label[i] == 2) and (current_set[4] == 1 or current_set[6] == 1 or current_set[7] == 1 or current_set[3] == 1 or current_set[5] == 1):
                overlap_count += 1
            
        elif (label[i] == 4 or label[i] == 6 or  label[i] == 7) and (current_set[0] == 1 or current_set[1] == 1 or current_set[2] or current_set[3] == 1 or current_set[5] == 1):
                overlap_count += 1

        elif (label[i] == 3 or label[i] == 5) and (current_set[4] == 1 or current_set[6] == 1 or current_set[7] == 1 or current_set[0] == 1 or current_set[1] == 1 or current_set[2] == 1):
                overlap_count += 1


    perecentage_of_overlap  = (overlap_count/conformal_set.shape[0])*100

    return perecentage_of_overlap




def breast_cancer_confusion_set_Overlap_metric(conformal_set, label):

    overlap_count = 0

    for i in range(conformal_set.shape[0]):
        current_set = conformal_set[i]

        if (label[i] == 1) and (current_set[2] == 1):
            overlap_count += 1

        elif (label[i] == 2) and (current_set[1] == 1):
            overlap_count += 1

        

        elif (label[i] == 4) and (current_set[5] == 1):
            overlap_count += 1

        elif (label[i] == 5) and (current_set[4] == 1):
            overlap_count += 1



    perecentage_of_overlap  = (overlap_count/conformal_set.shape[0])*100

    return perecentage_of_overlap
    







def confusion_set_Overlap_metric(conformal_set, label):

    overlap_count = 0

    for i in range(conformal_set.shape[0]):
        current_set = conformal_set[i]

        if (label[i] == 4) and (current_set[6] == 1):
            overlap_count += 1

        elif (label[i] == 6) and (current_set[4] == 1):
            overlap_count += 1

        
        elif (label[i] == 3) and (current_set[5] == 1):
            overlap_count += 1

        elif (label[i] == 5) and (current_set[3] == 1):
            overlap_count += 1

    perecentage_of_overlap  = (overlap_count/conformal_set.shape[0])*100

    return perecentage_of_overlap
    