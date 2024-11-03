import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from CP_methods import THR, APS, RAPS
import torch
from utils import avg_set_size_metric, coverage_gap_metric, breast_cancer_class_Overlap_metric, breast_cancer_confusion_set_Overlap_metric
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Trials', default=100, type=int, help= 'Number of total trials')
    parser.add_argument('--softmax_output_file_path', default='/path', type=str, help='path to the softmax_output_file')
    parser.add_argument('--expt_no', default=1, type=int, help= 'Expt no :-1, 2, 3, 4')
    parser.add_argument('--split', default=0.1, type=float, help='Calib/test split ratio')
    parser.add_argument('--CP_method', default='THR', type=str, help='CP method :- 1)THR  2)APS  3)RAPS')
    parser.add_argument('--alpha', default=0.1, type=float, help='value of alpha for CP coverage')
    parser.add_argument('--rand', default=True, type=bool, help='rand :- True/False for RAPS')
    parser.add_argument('--k_reg', default=2, type=int, help='value of k_reg for RAPS')
    parser.add_argument('--lambd', default=0.1, type=float, help='value of lambd for RAPS')
    args = parser.parse_args()


    avg_set_size_len_for_T_trials = []
    avg_coverage_gap_for_T_trials = []
    avg_coverage_for_T_trials = []

    normal_avg_set_size_len_for_T_trials = []
    abnormal_avg_set_size_len_for_T_trials = []


    perecentage_of_overlap_for_T_trials = []

    confusion_set_Overlap_metric_for_T_trials = []


    for t in range(args.Trials):
        print()
        print(f'Trials :- {t}')
        print()


        # loading the annotation file :-
        df = pd.read_csv(args.softmax_output_file_path)
        df = df.sample(frac=1).reset_index(drop=True)


        # calib-test split :- 
        feature_test, feature_calib = train_test_split(df, test_size = args.split, stratify=df['Label'], random_state=42)

        feature_test = feature_test.reset_index(drop=True)
        feature_calib = feature_calib.reset_index(drop=True)

        prob_output = feature_calib.iloc[:,:-1]
        df_np = prob_output.values 
        df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)

        prob_output = feature_test.iloc[:,:-1]
        df_np = prob_output.values
        df_prob_output_test = torch.tensor(df_np, dtype=torch.float32)


        true_class = feature_calib.iloc[:,-1]
        df_np = true_class.values
        df_true_class_calib = torch.tensor(df_np, dtype=torch.int)


        true_class = feature_test.iloc[:,-1]
        df_np = true_class.values
        df_true_class_test = torch.tensor(df_np, dtype=torch.int)





        if args.CP_method == 'THR':
            conformal_wrapper = THR(df_prob_output_calib, df_true_class_calib, args.alpha)
            quantile_value = conformal_wrapper.quantile()

            conformal_set = conformal_wrapper.prediction(df_prob_output_test, quantile_value)
           


            
        elif args.CP_method == 'APS':
            conformal_wrapper = APS(df_prob_output_calib, df_true_class_calib, args.alpha)
            quantile_value = conformal_wrapper.quantile()

            conformal_set = conformal_wrapper.prediction(df_prob_output_test, quantile_value)


        elif args.CP_method == 'RAPS':

            conformal_wrapper = RAPS(df_prob_output_calib, df_true_class_calib, args.alpha, args.k_reg, args.lambd, args.rand)
            quantile_value = conformal_wrapper.quantile()

            conformal_set = conformal_wrapper.prediction(df_prob_output_test, quantile_value)


        
        if args.expt_no == 1:

            avg_set_size = avg_set_size_metric(conformal_set)
            print(f'avg_set_size:- {avg_set_size}')
            
            coverage_gap, coverage = coverage_gap_metric(conformal_set, df_true_class_test, args.alpha)
            #print(f'coverage_gap:- {coverage_gap}')
            #print(f'coverage:- {coverage}')

            
            avg_set_size_len_for_T_trials.append(avg_set_size)
            avg_coverage_gap_for_T_trials.append(coverage_gap)
            avg_coverage_for_T_trials.append(coverage)

        
        elif args.expt_no == 2:
            label = df_true_class_test
            indices_0 = torch.nonzero(label == 0).squeeze()
            indices_1 = torch.nonzero(label == 1).squeeze()
            indices_2 = torch.nonzero(label == 2).squeeze()
            indices_3 = torch.nonzero(label == 3).squeeze()
            indices_4 = torch.nonzero(label == 4).squeeze()
            indices_5 = torch.nonzero(label == 5).squeeze()
            indices_6 = torch.nonzero(label == 6).squeeze()
            indices_7 = torch.nonzero(label == 7).squeeze()


            Normal_idx = torch.cat((indices_0, indices_1, indices_2, indices_3))
            Abnormal_idx = torch.cat((indices_4, indices_5, indices_6, indices_7))

            normal_conformal_prediction_set = conformal_set[Normal_idx, :]
            abnormal_conformal_prediction_set = conformal_set[Abnormal_idx, :]

            normal_avg_set_size_len = avg_set_size_metric(normal_conformal_prediction_set)
            abnormal_avg_set_size_len = avg_set_size_metric(abnormal_conformal_prediction_set)

            normal_avg_set_size_len_for_T_trials.append(normal_avg_set_size_len)
            abnormal_avg_set_size_len_for_T_trials.append(abnormal_avg_set_size_len)

        
        elif args.expt_no == 3:
            perecentage_of_overlap = breast_cancer_class_Overlap_metric(conformal_set, df_true_class_test)
            
            perecentage_of_overlap_for_T_trials.append(perecentage_of_overlap)



        elif args.expt_no == 4:
            perecentage_of_confusion = breast_cancer_confusion_set_Overlap_metric(conformal_set, df_true_class_test)
            #print(f'perecentage_of_confusion :- {perecentage_of_confusion}')

            confusion_set_Overlap_metric_for_T_trials.append(perecentage_of_confusion)












    
    if args.expt_no == 1:
        avg_set_size_len_for_T_trials = np.array(avg_set_size_len_for_T_trials)
        average = np.mean(avg_set_size_len_for_T_trials)
        std_dev = np.std(avg_set_size_len_for_T_trials, ddof=1)

        print()
        print()
        print()
        print()
        print(f"Average set_size_len_for_T_trials: {average}")
        print(f"Standard Deviation set_size_len_for_T_trials: {std_dev}")

    elif args.expt_no == 2:
        print()
        print()
        print(f'set_size :-')

        normal_avg_set_size_len_for_T_trials = np.array(normal_avg_set_size_len_for_T_trials)
        normal_average_set_size_len = np.mean(normal_avg_set_size_len_for_T_trials)
        normal_std_dev_set_size_len = np.std(normal_avg_set_size_len_for_T_trials, ddof=1)

        print()
        print(f"normal_average_set_size_len: {normal_average_set_size_len}")
        print(f"normal_std_dev_set_size_len: {normal_std_dev_set_size_len}")



        abnormal_avg_set_size_len_for_T_trials = np.array(abnormal_avg_set_size_len_for_T_trials)
        abnormal_average_set_size_len = np.mean(abnormal_avg_set_size_len_for_T_trials)
        abnormal_std_dev_set_size_len = np.std(abnormal_avg_set_size_len_for_T_trials, ddof=1)

        print()
        print(f"abnormal_average_set_size_len: {abnormal_average_set_size_len}")
        print(f"abnormal_std_dev_set_size_len: {abnormal_std_dev_set_size_len}")


    elif args.expt_no == 3:
        perecentage_of_overlap_for_T_trials = np.array(perecentage_of_overlap_for_T_trials)
        average_perecentage_of_overlap_for_T_trials = np.mean(perecentage_of_overlap_for_T_trials)
        std_dev_perecentage_of_overlap_for_T_trials = np.std(perecentage_of_overlap_for_T_trials, ddof=1)

        print()
        print(f"average_perecentage_of_overlap_for_T_trials: {average_perecentage_of_overlap_for_T_trials}")
        print(f"std_dev_perecentage_of_overlap_for_T_trials: {std_dev_perecentage_of_overlap_for_T_trials}")


    elif args.expt_no == 4:
        confusion_set_Overlap_metric_for_T_trials = np.array(confusion_set_Overlap_metric_for_T_trials)
        average_confusion_set_Overlap_metric_for_T_trials = np.mean(confusion_set_Overlap_metric_for_T_trials)
        std_dev_confusion_set_Overlap_metric_for_T_trials = np.std(confusion_set_Overlap_metric_for_T_trials, ddof=1)

        print()
        print(f"average_confusion_set_Overlap_metric_for_T_trials: {average_confusion_set_Overlap_metric_for_T_trials}")
        print(f"std_dev_confusion_set_Overlap_metric_for_T_trials: {std_dev_confusion_set_Overlap_metric_for_T_trials}")



    


if __name__ == '__main__':
    main()