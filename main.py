import argparse
import os
from dataset import cytology_dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import engine 
from utils import Multiclass_classification_metrices
from model import CustomResNet50, CustomResNet152, CustomEfficientNet, CustomXception, CustomDenseNet, CustomViT, CustomRegNet, CustomResNeXt, Custom_DieT

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=8, type=int, help='  CRIC:- 8 classes')
    parser.add_argument('--num_epochs', default=100, type=int, help= 'Number of total training epochs')
    parser.add_argument('--img_dir', default='/path', type=str, help='path to the image directory')
    parser.add_argument('--model', default='CustomResNet50', type=str, help='Model to be used')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size for the dataloader')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers for dataloader')
    parser.add_argument('--lr', default=0.0002, type=float, help='Initial learning rate')
    parser.add_argument('--weight_decay', default=5e-3, type=float, help='Weight Decay')
    parser.add_argument('--patience', default=20, type=int, help='Representing the number of consecutive epochs where the performance metric does not improve before training stops')
    parser.add_argument('--folds', default=5, type=int, help='No of folds in K-folds')
    parser.add_argument('--loss', default='cross_entropy', type=str, help='Loss :- 1)cross_entropy  2)hinge_loss')
    parser.add_argument('--percentage_change', default=20, type=float, help='percentage_change')
    parser.add_argument('--metric', default='class_Overlap_metric', type=str, help='metric :- 1)class_Overlap_metric  2)confusion_set_Overlap_metric  3)expt2')
                        
    args = parser.parse_args()

    

    for fold in range(args.folds):
        print(f'folde :- {fold}')

        
        path = '/kaggle/working/'

        train = str(fold) + 'train.csv'
        train_path = os.path.join(path, 'Train_Val_split', train)
        
        train_dataset = cytology_dataset(img_dir = args.img_dir, annotation_file = train_path, img_transform = True)
        train_loader = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)

        val = str(fold) + 'val.csv'
        val_path = os.path.join(path, 'Train_Val_split', val)
        
        val_dataset = cytology_dataset(img_dir = args.img_dir, annotation_file = val_path, img_transform = True)
        val_loader = DataLoader(val_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)


        test_path = os.path.join(path, 'Test', 'test.csv')
        
        test_dataset = cytology_dataset(img_dir = args.img_dir, annotation_file = test_path, img_transform = True)
        test_loader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.num_workers)
        

        # model :- 
        if args.model == 'CustomResNet50':
            model = CustomResNet50(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'CustomResNet152':
            model = CustomResNet152(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'CustomViT':
            model = CustomViT(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'Custom_DieT':
            model = Custom_DieT(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'CustomResNeXt':
            model = CustomResNeXt(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'CustomEfficientNet':
            model = CustomEfficientNet(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'CustomXception':
            model = CustomXception(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        elif args.model == 'CustomDenseNet':
            model = CustomDenseNet(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        

        elif args.model == 'CustomRegNet':
            model = CustomRegNet(args.num_classes)
            device = torch.device("cuda")
            model.to(device)

        

        


        # set up the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.99))

        patience = args.patience
        best_val_loss = np.inf
        epochs_without_improvement = 0
        best_model_state = None
        
        for epoch in range(args.num_epochs):
            print(f'epoch :-{epoch}')

            engine.train(train_loader, model, optimizer, args.loss, args.metric, args.percentage_change)

            _, _, val_loss= engine.val(val_loader, model, args.loss, args.metric, args.percentage_change)
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
        val_predictions, val_labels, _ = engine.val(val_loader, model, args.loss, args.metric, args.percentage_change)
        val_auc, val_acc = Multiclass_classification_metrices(val_labels, val_predictions, args.num_classes)
        print(f'val_auc:-{val_auc}')
        print(f'val_acc:-{val_acc}')
        print()



        
        print()
        print('test')
        test_predictions, test_labels, softmax_values_list, df_combined = engine.test(test_loader, model)
        output = str(fold) + 'softmax_output.csv'
        df_combined.to_csv(os.path.join(path, output), index=False)
        test_auc, test_acc = Multiclass_classification_metrices(test_labels, test_predictions, args.num_classes)
        print(f'test_auc:-{test_auc}')
        print(f'test_acc:-{test_acc}')
        print()


        


if __name__ == '__main__':
    main()
    