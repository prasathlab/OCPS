import argparse
import os
from dataset import feature_dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import engine 
from utils import Multiclass_classification_metrices
from model import Feature_FC_layer_for_Diet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=8, type=int, help='  CRIC:- 8 classes, Breast_Cancer datset :- 8 classes')
    parser.add_argument('--feat_dir', default='/path', type=str, help='path to the feature directory')
    parser.add_argument('--num_epochs', default=100, type=int, help= 'Number of total training epochs')
    parser.add_argument('--model', default='Feature_FC_layer_for_Diet', type=str, help='Model to be used')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size for the dataloader')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers for dataloader')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight Decay')
    parser.add_argument('--patience', default=20, type=int, help='Representing the number of consecutive epochs where the performance metric does not improve before training stops')
    parser.add_argument('--folds', default=5, type=int, help='No of folds in K-folds')
    parser.add_argument('--loss', default='cross_entropy', type=str, help='Loss :- 1)cross_entropy  2)hinge_loss')
    parser.add_argument('--dataset', default='Breast_cancer', type=str, help='dataset :- 1)Breast_cancer  2)Cervical_cancer')
    parser.add_argument('--metric', default='class_Overlap_metric', type=str, help='metric :- 1)class_Overlap_metric  2)confusion_set_Overlap_metric  3)expt2')
    parser.add_argument('--distance_in_hinge_loss', default=5, type=float, help='distance_in_hinge_loss')


    args = parser.parse_args()

    path = '/kaggle/working/'
    
    #path = '/kaggle/input/cp-label-5fold-dataset/CP_dataset/'

    df_feature = pd.read_csv(args.feat_dir)


    for fold in range(args.folds):
        print(f'folde :- {fold}')


        

        train = str(fold) + 'train.csv'
        train_path = os.path.join(path, 'Train_Val_split', train)
        df_train = pd.read_csv(train_path)

        train_dataset = feature_dataset(df_feature = df_feature, df_label = df_train)
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)


        val = str(fold) + 'val.csv'
        val_path = os.path.join(path, 'Train_Val_split', val)
        df_val = pd.read_csv(val_path)
        
        val_dataset = feature_dataset(df_feature = df_feature, df_label = df_val)
        val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)



        test_path = os.path.join(path, 'Test', 'test.csv')
        df_test = pd.read_csv(test_path)
        
        test_dataset = feature_dataset(df_feature = df_feature, df_label = df_test)
        test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)


        # model :- 
        if args.model == 'Feature_FC_layer_for_Diet':
            model = Feature_FC_layer_for_Diet(args.num_classes)
            device = torch.device("cuda")
            model.to(device)


        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.99))

        patience = args.patience
        best_val_loss = np.inf
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(args.num_epochs):
            print(f'epoch :-{epoch}')

            engine.train(train_loader, model, optimizer, args.loss, args.dataset, args.metric, args.distance_in_hinge_loss)

            _, _, val_loss= engine.val(val_loader, model, args.loss, args.dataset, args.metric, args.distance_in_hinge_loss)
            print(f'val_loss:-{val_loss}')


            # early stopping :- 
            if epoch>=10:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    # Save the state dictionary of the best model
                    best_model_state = model.state_dict()

                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print(f"Early stopping after {epoch+1} epochs without improvement.")
                        break


        # Load the best model state dictionary for val metrices :-
        if best_model_state is not None:
            model.load_state_dict(best_model_state)



        print()
        print('val')
        val_predictions, val_labels, _ = engine.val(val_loader, model, args.loss, args.dataset, args.metric, args.distance_in_hinge_loss)
        val_auc, val_acc = Multiclass_classification_metrices(val_labels, val_predictions, args.num_classes)
        print(f'val_auc:-{val_auc}')
        print(f'val_acc:-{val_acc}')
        print()


        print()
        print('test')
        test_predictions, test_labels, softmax_values_list, df_combined = engine.test(test_loader, model)
        if args.loss == 'hinge_loss':
            output = str(args.distance_in_hinge_loss) + '_' + str(fold) + '_' + 'softmax_output.csv'

        else:
            output = str(fold) + 'softmax_output.csv'
        df_combined.to_csv(os.path.join('/kaggle/working/', output), index=False)
        test_auc, test_acc = Multiclass_classification_metrices(test_labels, test_predictions, args.num_classes)
        print(f'test_auc:-{test_auc}')
        print(f'test_acc:-{test_acc}')
        print()




if __name__ == '__main__':
    main()