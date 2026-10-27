import torch.nn as nn

class CNNClassifier(nn.Module):
    def __init__(self, model_name = 'CNN_generic', num_classes = 10, in_channels = 3, dropout = 0.5):
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.dropout = dropout

        # CNN con Sequential
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # Fully connected finale
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.mean(dim=[2,3])

        x = self.classifier(x)
        return x

    def print_model(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"\n{'MODEL SUMMARY':=^70}")
        print(f"Model name      : {self.model_name}")
        print(f"Input channels  : {self.in_channels}")
        print(f"Num classes     : {self.num_classes}")
        print(f"Dropout         : {self.dropout}")

        print(f"Total parameters     : {total_params:,}")
        print(f"Trainable parameters : {trainable_params:,}")
        print("=" * 70)

