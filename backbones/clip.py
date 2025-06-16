import open_clip
import torch.nn as nn

class ResNet50_oCLIP(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50_oCLIP, self).__init__()
        model, _, _ = open_clip.create_model_and_transforms(
            'RN50',
            pretrained='openai' if pretrained else None
        )
        self.backbone = model.visual  # 
        
    def forward(self, x):
        # Run up to the final residual block
        x = self.backbone.stem(x)
        x = self.backbone.layer1(x)  # output x2
        x = self.backbone.layer2(x)  # output x3
        x = self.backbone.layer3(x)  # output x4
        x = self.backbone.layer4(x)  # output x5
        return x  # This will be [B, 2048, 7, 7]
        

def resnet50_oCLIP(pretrained=True):
    return ResNet50_oCLIP(pretrained=pretrained)


    

if __name__ == "__main__":
    import torch

    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images  image size 224*224
 
    # Load model
    model = resnet50_oCLIP(pretrained=True)
    model.eval()

    with torch.no_grad():
        features = model(dummy_input)

    print("oCLIP ResNet50 feature shape:", features.shape)