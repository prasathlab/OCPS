import argparse
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import os



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/path', type=str, help='path to the img feature file')
    parser.add_argument('--split', default=0.2, type=float, help='Train/Test split ratio')
    parser.add_argument('--folds', default=5, type=int, help='No of folds in K-folds')
    args = parser.parse_args()


    # loading the annotation file :-
    df = pd.read_csv(args.dataset)
    df = df.sample(frac=1).reset_index(drop=True)

    path_train, path_test, label_train, label_test, feature_train, feature_test  = train_test_split(df['path'], df['label'], df.iloc[:, 2:],   test_size = args.split, stratify=df['label'], random_state=42)
    path_train = path_train.to_frame()
    path_test = path_test.to_frame()
    label_train = label_train.to_frame()
    label_test = label_test.to_frame()
    df_train = path_train.join(label_train).reset_index(drop=True)
    df_test = path_test.join(label_test).reset_index(drop=True)
    
    
    path = '/kaggle/working/'
    os.makedirs(os.path.join(path, 'Test'), exist_ok=True)
    df_test.to_csv(os.path.join(path, 'Test', 'test.csv'), index=False)


    kf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    i=0
    os.makedirs(os.path.join(path, 'Train_Val_split'), exist_ok=True)




    for train_idx, val_idx in kf.split(df_train['path'], df_train['label']):
        train_df = df_train.iloc[train_idx].reset_index(drop=True)
        train = str(i) + 'train.csv'
        train_df.to_csv(os.path.join(path, 'Train_Val_split', train), index=False)

        df_val = df_train.iloc[val_idx].reset_index(drop=True)
        val = str(i) + 'val.csv'
        df_val.to_csv(os.path.join(path, 'Train_Val_split', val), index=False)
        i = i + 1



if __name__ == '__main__':
    main()
