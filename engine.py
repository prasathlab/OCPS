import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from utils import hinge_loss






def train(data_loader, model, optimizer, Loss, dataset, metric, percentage_change):

    # put the model in train mode
    model.train()

    for data in data_loader:
        feature = data[0].float()
        label = data[1]

        # Check if CUDA is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")


        # Move the tensor to the selected device (CPU or CUDA)
        feature = feature.to(device)
        label = label.to(device)

        outputs = model(feature)


        if Loss == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, label)

        elif Loss == 'hinge_loss':
            loss = hinge_loss(metric, outputs, label, percentage_change, dataset)

        # zero grad the optimizer
        optimizer.zero_grad()

        # calculate the gradient
        loss.backward()

        # update the weights
        optimizer.step()

        torch.cuda.empty_cache()








def val(data_loader, model, Loss, dataset, metric, percentage_change):
    val_loss_list = []
    final_output = []
    final_label = []

    # put model in evaluation mode
    model.eval()

    with torch.no_grad():
        for data in data_loader:
            feature = data[0].float()
            label = data[1]

            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

             # Move the tensor to the selected device (CPU or CUDA)
            feature = feature.to(device)
            label = label.to(device)

            outputs = model(feature)


            if Loss == 'cross_entropy':
                criterion = nn.CrossEntropyLoss()
                temp_val_loss = criterion(outputs, label)
                val_loss_list.append(temp_val_loss)
                softmax_values = F.softmax(outputs, dim=1)
                outputs = torch.argmax(softmax_values, dim=1).int()

            elif Loss == 'hinge_loss':
                temp_val_loss = hinge_loss(metric, outputs, label, percentage_change, dataset)
                val_loss_list.append(temp_val_loss)
                softmax_values = F.softmax(outputs, dim=1)
                outputs = torch.argmax(softmax_values, dim=1).int()


            OUTPUTS = outputs.detach().cpu().tolist()
            final_output.extend(OUTPUTS)
            final_label.extend(label.detach().cpu().tolist())

            torch.cuda.empty_cache()


    return final_output, final_label, sum(val_loss_list)/len(val_loss_list)









def test(data_loader, model):
    final_output = []
    final_label = []
    softmax_values_list = []

    # put model in evaluation mode
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            feature = data[0].float()
            label = data[1]

            # Check if CUDA is available
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

             # Move the tensor to the selected device (CPU or CUDA)
            feature = feature.to(device)
            label = label.to(device)

            outputs = model(feature)
            softmax_values = F.softmax(outputs, dim=1)
            softmax_values_list.extend(softmax_values.detach().cpu())
            outputs = torch.argmax(softmax_values, dim=1).int()
            OUTPUTS = outputs.detach().cpu().tolist()

            final_output.extend(OUTPUTS)
            final_label.extend(label.detach().cpu().tolist())

            torch.cuda.empty_cache()


    # Convert final_label and softmax_values_list to individual DataFrames
    df_labels = pd.DataFrame(final_label, columns=['Label'])
    numpy_array = np.stack([t.numpy() for t in softmax_values_list])
    df_softmax = pd.DataFrame(numpy_array)
    #df_softmax = pd.DataFrame(softmax_values_list, columns=[f'Class_{i}' for i in range(len(softmax_values_list[0]))])

    # Concatenate DataFrames column-wise
    df_combined = pd.concat([df_softmax, df_labels], axis=1)
    
    
    

    
    return final_output, final_label, softmax_values_list, df_combined


            

    






        
