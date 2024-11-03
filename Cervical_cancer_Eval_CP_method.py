import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from CP_methods import THR, APS, RAPS
import torch
from utils import avg_set_size_metric, coverage_gap_metric, class_Overlap_metric, confusion_set_Overlap_metric
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Trials', default=100, type=int, help= 'Number of total trials for eval CP method')
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
    normal_avg_coverage_gap_for_T_trials = []
    normal_avg_coverage_for_T_trials = []

    LG_avg_set_size_len_for_T_trials = []
    LG_avg_coverage_gap_for_T_trials = []
    LG_avg_coverage_for_T_trials = []

    HG_avg_set_size_len_for_T_trials = []
    HG_avg_coverage_gap_for_T_trials = []
    HG_avg_coverage_for_T_trials = []



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
        test_0, calib_0, test_1, calib_1, test_2, calib_2, test_3, calib_3, test_4, calib_4, test_5, calib_5, test_6, calib_6, test_7, calib_7, test_label, calib_label = train_test_split(df['0'],df['1'], df['2'],df['3'], df['4'],df['5'], df['6'],df['7'], df['Label'], test_size = 0.1, stratify=df['Label'], random_state=42)

        test_0 = test_0.to_frame()
        test_1 = test_1.to_frame()
        test_2 = test_2.to_frame()
        test_3 = test_3.to_frame()
        test_4 = test_4.to_frame()
        test_5 = test_5.to_frame()
        test_6 = test_6.to_frame()
        test_7 = test_7.to_frame()
        test = test_0.join(test_1, how='inner').join(test_2, how='inner').join(test_3, how='inner').join(test_4, how='inner').join(test_5, how='inner').join(test_6, how='inner').join(test_7, how='inner').join(test_label, how='inner')
        test = test.reset_index(drop=True)

        calib_0 = calib_0.to_frame()
        calib_1 = calib_1.to_frame()
        calib_2 = calib_2.to_frame()
        calib_3 = calib_3.to_frame()
        calib_4 = calib_4.to_frame()
        calib_5 = calib_5.to_frame()
        calib_6 = calib_6.to_frame()
        calib_7 = calib_7.to_frame()
        calib = calib_0.join(calib_1, how='inner').join(calib_2, how='inner').join(calib_3, how='inner').join(calib_4, how='inner').join(calib_5, how='inner').join(calib_6, how='inner').join(calib_7, how='inner').join(calib_label, how='inner')
        calib = calib.reset_index(drop=True)

        prob_output = calib.iloc[:,:-1]
        df_np = prob_output.values 
        df_prob_output_calib = torch.tensor(df_np, dtype=torch.float32)

        prob_output = test.iloc[:,:-1]
        df_np = prob_output.values
        df_prob_output_test = torch.tensor(df_np, dtype=torch.float32)



        true_class = calib.iloc[:,-1]
        df_np = true_class.values
        df_true_class_calib = torch.tensor(df_np, dtype=torch.int)


        true_class = test.iloc[:,-1]
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
            #print(f'avg_set_size:- {avg_set_size}')
            
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


            Normal_idx = torch.cat((indices_0, indices_1, indices_2))
            LG_idx = torch.cat((indices_3, indices_5))
            HG_idx = torch.cat((indices_4, indices_6, indices_7))

            normal_conformal_prediction_set = conformal_set[Normal_idx, :]
            LG_conformal_prediction_set = conformal_set[LG_idx, :]
            HG_conformal_prediction_set = conformal_set[HG_idx, :]


            normal_avg_set_size_len = avg_set_size_metric(normal_conformal_prediction_set)
            LG_avg_set_size_len = avg_set_size_metric(LG_conformal_prediction_set)
            HG_avg_set_size_len = avg_set_size_metric(HG_conformal_prediction_set)
            """
            
            print()
            print(f'normal_avg_set_size_len:- {normal_avg_set_size_len}')
            print(f'LG_avg_set_size_len:- {LG_avg_set_size_len}')
            print(f'HG_avg_set_size_len:- {HG_avg_set_size_len}')
            print()
            """

        
            normal_true_class = df_true_class_test[Normal_idx]
            LG_true_class = df_true_class_test[LG_idx]
            HG_true_class = df_true_class_test[HG_idx]


            normal_coverage_gap, normal_coverage = coverage_gap_metric(normal_conformal_prediction_set, normal_true_class, args.alpha)
            LG_coverage_gap, LG_coverage = coverage_gap_metric(LG_conformal_prediction_set, LG_true_class, args.alpha)
            HG_coverage_gap, HG_coverage = coverage_gap_metric(HG_conformal_prediction_set, HG_true_class, args.alpha)
            """
            print()
            print(f'normal_coverage_gap:- {normal_coverage_gap}')
            print(f'normal_coverage:- {normal_coverage}')
            print(f'LG_coverage_gap:- {LG_coverage_gap}')
            print(f'LG_coverage:- {LG_coverage}')
            print(f'HG_coverage_gap:- {HG_coverage_gap}')
            print(f'HG_coverage_gap:- {HG_coverage_gap}')
            print()
            """




            normal_avg_set_size_len_for_T_trials.append(normal_avg_set_size_len)
            normal_avg_coverage_gap_for_T_trials.append(normal_coverage_gap)
            normal_avg_coverage_for_T_trials.append(normal_coverage)

            LG_avg_set_size_len_for_T_trials.append(LG_avg_set_size_len)
            LG_avg_coverage_gap_for_T_trials.append(LG_coverage_gap)
            LG_avg_coverage_for_T_trials.append(LG_coverage)

            HG_avg_set_size_len_for_T_trials.append(HG_avg_set_size_len)
            HG_avg_coverage_gap_for_T_trials.append(HG_coverage_gap)
            HG_avg_coverage_for_T_trials.append(HG_coverage)


        
        elif args.expt_no == 3:
            perecentage_of_overlap = class_Overlap_metric(conformal_set, df_true_class_test)
            #print(f'perecentage_of_overlap :- {perecentage_of_overlap}')

            perecentage_of_overlap_for_T_trials.append(perecentage_of_overlap)



        elif args.expt_no == 4:
            perecentage_of_confusion = confusion_set_Overlap_metric(conformal_set, df_true_class_test)
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




        avg_coverage_gap_for_T_trials = np.array(avg_coverage_gap_for_T_trials)
        average = np.mean(avg_coverage_gap_for_T_trials)
        std_dev = np.std(avg_coverage_gap_for_T_trials, ddof=1)

        print()
        print(f"Average coverage_gap_for_T_trials: {average}")
        print(f"Standard Deviation coverage_gap_for_T_trials: {std_dev}")




        avg_coverage_for_T_trials = np.array(avg_coverage_for_T_trials)
        average = np.mean(avg_coverage_for_T_trials)
        std_dev = np.std(avg_coverage_for_T_trials, ddof=1)

        print()
        print(f"Average coverage_for_T_trials: {average}")
        print(f"Standard Deviation coverage_for_T_trials: {std_dev}")



    
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

        LG_avg_set_size_len_for_T_trials = np.array(LG_avg_set_size_len_for_T_trials)
        LG_average_set_size_len = np.mean(LG_avg_set_size_len_for_T_trials)
        LG_std_dev_set_size_len = np.std(LG_avg_set_size_len_for_T_trials, ddof=1)

        print()
        print(f"LG_average_set_size_len: {LG_average_set_size_len}")
        print(f"LG_std_dev_set_size_len: {LG_std_dev_set_size_len}")


        HG_avg_set_size_len_for_T_trials = np.array(HG_avg_set_size_len_for_T_trials)
        HG_average_set_size_len = np.mean(HG_avg_set_size_len_for_T_trials)
        HG_std_dev_set_size_len = np.std(HG_avg_set_size_len_for_T_trials, ddof=1)

        print()
        print(f"HG_average_set_size_len: {HG_average_set_size_len}")
        print(f"HG_std_dev_set_size_len: {HG_std_dev_set_size_len}")








        print()
        print()
        print(f'coverage_gap :-')
        normal_avg_coverage_gap_for_T_trials = np.array(normal_avg_coverage_gap_for_T_trials)
        normal_avg_coverage_gap = np.mean(normal_avg_coverage_gap_for_T_trials)
        normal_std_dev_coverage_gap = np.std(normal_avg_coverage_gap_for_T_trials)

        print()
        print(f"normal_avg_coverage_gap: {normal_avg_coverage_gap}")
        print(f"normal_std_dev_coverage_gap: {normal_std_dev_coverage_gap}")



        LG_avg_coverage_gap_for_T_trials = np.array(LG_avg_coverage_gap_for_T_trials)
        LG_avg_coverage_gap = np.mean(LG_avg_coverage_gap_for_T_trials)
        LG_std_dev_coverage_gap = np.std(LG_avg_coverage_gap_for_T_trials)

        print()
        print(f"LG_avg_coverage_gap: {LG_avg_coverage_gap}")
        print(f"LG_std_dev_coverage_gap: {LG_std_dev_coverage_gap}")


        HG_avg_coverage_gap_for_T_trials = np.array(HG_avg_coverage_gap_for_T_trials)
        HG_avg_coverage_gap = np.mean(HG_avg_coverage_gap_for_T_trials)
        HG_std_dev_coverage_gap = np.std(HG_avg_coverage_gap_for_T_trials)

        print()
        print(f"HG_avg_coverage_gap: {HG_avg_coverage_gap}")
        print(f"HG_std_dev_coverage_gap: {HG_std_dev_coverage_gap}")








        print()
        print()
        print(f'coverage:-')
        normal_avg_coverage_for_T_trials = np.array(normal_avg_coverage_for_T_trials)
        normal_average_coverage = np.mean(normal_avg_coverage_for_T_trials)
        normal_std_dev_coverage = np.std(normal_avg_coverage_for_T_trials, ddof=1)

        print()
        print(f"normal_average_coverage: {normal_average_coverage}")
        print(f"normal_std_dev_coverage: {normal_std_dev_coverage}")


        LG_avg_coverage_for_T_trials = np.array(LG_avg_coverage_for_T_trials)
        LG_average_coverage = np.mean(LG_avg_coverage_for_T_trials)
        LG_std_dev_coverage = np.std(LG_avg_coverage_for_T_trials, ddof=1)

        print()
        print(f"LG_average_coverage: {LG_average_coverage}")
        print(f"LG_std_dev_coverage: {LG_std_dev_coverage}")


        HG_avg_coverage_for_T_trials = np.array(HG_avg_coverage_for_T_trials)
        HG_average_coverage = np.mean(HG_avg_coverage_for_T_trials)
        HG_std_dev_coverage = np.std(HG_avg_coverage_for_T_trials, ddof=1)

        print()
        print(f"HG_average_coverage: {HG_average_coverage}")
        print(f"HG_std_dev_coverage: {HG_std_dev_coverage}")



    
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