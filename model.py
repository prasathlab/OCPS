import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
from transformers import DeiTModel, DeiTConfig, ViTModel, ViTConfig
import timm



class CustomResNet50(nn.Module):
    def __init__(self, num_class):
        super(CustomResNet50, self).__init__()
        # Load pre-trained ResNet50
        resnet = models.resnet50(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Add a new fully connected layer 
        self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )

    def forward(self, x):
        # Forward pass through ResNet base
        x = self.resnet(x)
        # Flatten features
        x = torch.flatten(x, 1)
        # Final FC layer for classification (logits)
        x = self.fc(x)
        return x
    

class CustomResNet152(nn.Module):
    def __init__(self, num_class):
        super(CustomResNet152, self).__init__()
        # Load pre-trained ResNet152
        resnet = models.resnet152(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnet.parameters():
            param.requires_grad = False
        # Remove the final fully connected layer
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        # Add a new fully connected layer 
        self.fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )

    def forward(self, x):
        # Forward pass through ResNet base
        x = self.resnet(x)
        # Flatten features
        x = torch.flatten(x, 1)
        # Final FC layer for classification (logits)
        x = self.fc(x)
        return x
    
    

class Custom_DieT(nn.Module):
    def __init__(self, num_class):
        super(Custom_DieT, self).__init__()
        self.num_class = num_class

        # Define ViT model
        config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.ViT = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
        for param in self.ViT.parameters():
            param.requires_grad = False

        self.ViT_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_class)
            )

    
    def forward(self, x):
        batch_size, _, _, _ = x.shape
        x2 = self.ViT(x).last_hidden_state[:, 0, :]
        x2 = self.ViT_fc(x2)
        return x2
    




class Feature_Extractor_Diet(nn.Module):
    def __init__(self):
        super(Feature_Extractor_Diet, self).__init__()
        

        # Define ViT model
        config = DeiTConfig.from_pretrained("facebook/deit-base-distilled-patch16-224")
        self.ViT = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224", config=config)
        for param in self.ViT.parameters():
            param.requires_grad = False

    def forward(self, x):
        
        x2 = self.ViT(x).last_hidden_state[:, 0, :]
        
        return x2
    



class Feature_FC_layer_for_Diet(nn.Module):
    def __init__(self, num_classes):
        super(Feature_FC_layer_for_Diet, self).__init__()

        self.Diet_fc = nn.Sequential(
                nn.Linear(768, 512),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(p=0.3),
                nn.Linear(128, num_classes)
            )
        
    
    def forward(self, x):
        x1 = self.Diet_fc(x)
        return x1
        









    


class CustomViT(nn.Module):
    def __init__(self, num_classes):
        super(CustomViT, self).__init__()
        self.num_classes = num_classes

        # Define ViT model
        config = ViTConfig.from_pretrained("google/vit-base-patch16-224")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224", config=config)
        
        # Freeze all layers except the final fully connected layer
        for param in self.vit.parameters():
            param.requires_grad = False

        # Define custom fully connected layers for classification
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Forward pass through ViT model
        outputs = self.vit(x)
        
        # Extract [CLS] token representation (CLS token is at index 0)
        cls_token = outputs.last_hidden_state[:, 0, :]
        
        # Forward pass through custom fully connected layers for classification
        x = self.fc(cls_token)
        
        return x
    


class CustomResNeXt(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNeXt, self).__init__()
        # Load pre-trained ResNeXt50
        resnext = models.resnext50_32x4d(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in resnext.parameters():
            param.requires_grad = False
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(resnext.children())[:-2])
        # Add adaptive average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Add a new fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(resnext.fc.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Forward pass through ResNeXt base
        x = self.features(x)
        # Apply adaptive average pooling
        x = self.avgpool(x)
        # Flatten features
        x = torch.flatten(x, 1)
        # Final FC layer for classification (logits)
        x = self.classifier(x)
        return x
    



class CustomEfficientNet(nn.Module):
    def __init__(self, num_class):
        super(CustomEfficientNet, self).__init__()
        # Load pre-trained EfficientNet-B7
        efficientnet = models.efficientnet_b7(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in efficientnet.parameters():
            param.requires_grad = False
        # Extract features (excluding the classifier)
        self.features = efficientnet.features
        self.avgpool = efficientnet.avgpool
        # Add a new fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(efficientnet.classifier[1].in_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        # Forward pass through EfficientNet base
        x = self.features(x)
        # Global Average Pooling
        x = self.avgpool(x)
        # Flatten features
        x = torch.flatten(x, 1)
        # Final FC layer for classification (logits)
        x = self.classifier(x)
        return x
    



class CustomXception(nn.Module):
    def __init__(self, num_class):
        super(CustomXception, self).__init__()
        # Load pre-trained Xception
        xception = timm.create_model('xception', pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in xception.parameters():
            param.requires_grad = False
        # Extract features (excluding the classifier)
        self.features = nn.Sequential(*list(xception.children())[:-1])
        # Add a new fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(xception.num_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        # Forward pass through Xception base
        x = self.features(x)
        # Global Average Pooling
        #x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # Flatten features
        x = torch.flatten(x, 1)
        # Final FC layer for classification (logits)
        x = self.classifier(x)
        return x








class CustomDenseNet(nn.Module):
    def __init__(self, num_class):
        super(CustomDenseNet, self).__init__()
        # Load pre-trained DenseNet121
        densenet = models.densenet121(pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in densenet.parameters():
            param.requires_grad = False
        # Extract features (excluding the classifier)
        self.features = densenet.features
        # Add a new fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(densenet.classifier.in_features, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_class)
        )

    def forward(self, x):
        # Forward pass through DenseNet base
        x = self.features(x)
        # Global Average Pooling
        x = torch.nn.functional.relu(x, inplace=True)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        # Flatten features
        x = torch.flatten(x, 1)
        # Final FC layer for classification (logits)
        x = self.classifier(x)
        return x
        




class CustomRegNet(nn.Module):
    def __init__(self, num_classes):
        super(CustomRegNet, self).__init__()
        # Load pre-trained RegNetY-16GF model
        regnet = timm.create_model('regnetx_002', pretrained=True)
        # Freeze all layers except the final fully connected layer
        for param in regnet.parameters():
            param.requires_grad = False
        # Extract features (excluding the classifier)
        self.features = regnet
        self.features.head = nn.Identity()  # Remove the original classification head
        # Add a new fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(368, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Forward pass through RegNet base
        x = self.features(x)
        
        # Flatten features (if needed)
        if x.dim() > 2:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            
            x = x.flatten(1)
            
        # Final FC layer for classification (logits)
        x = self.classifier(x)
        return x
    




    





