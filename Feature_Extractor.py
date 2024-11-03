import argparse
import os
from dataset import feature_extraction_dataset 
import torch
from torch.utils.data import DataLoader
from model import Feature_Extractor_Diet
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default='/path', type=str, help='path to the image directory')
    parser.add_argument('--csv_file', default='/path', type=str, help='path to the csv file ')
    parser.add_argument('--model', default='Feature_Extractor_Diet', type=str, help='Model to be used for feature extraction')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size for the dataloader')
    parser.add_argument('--num_workers', default=4, type=int, help='num workers for dataloader')
    args = parser.parse_args()


    dataset = feature_extraction_dataset(img_dir = args.img_dir, annotation_file = args.csv_file, img_transform = True)
    data_loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = False, num_workers = args.num_workers)


    if args.model == 'Feature_Extractor_Diet':
        model = Feature_Extractor_Diet()
        device = torch.device("cuda")
        model.to(device)
    

    features_list = []
    path_list = []
    label_list = []
    i = 0
    for data in data_loader:
        i += 1
        print(i)
        feature = data[0].float()
        img_path = data[1]
        label = data[2]

         # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")


        # Move the tensor to the selected device (CPU or CUDA)
        feature = feature.to(device)

        img_features = model(feature)

        features_list.extend(img_features.detach().cpu())
        path_list.extend(img_path)
        label_list.extend(label.detach().cpu().tolist())

        torch.cuda.empty_cache()

        

        


   
    numpy_array = np.stack([t.numpy() for t in features_list])
    df_features = pd.DataFrame(numpy_array)

    df_path = pd.DataFrame(path_list, columns=['path'])

    df_label = pd.DataFrame(label_list, columns=['label'])

    df_combined = pd.concat([df_path, df_label, df_features], axis=1)


    path = '/kaggle/working/'
    output = 'features.csv'
    df_combined.to_csv(os.path.join(path, output), index=False)






if __name__ == '__main__':
    main()