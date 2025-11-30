import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import json


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        vgg16_config_path,
        vgg_backbone="VGG16",
        content_layers=["relu3_3"],
        style_layers=["relu1_2", "relu2_2", "relu3_3", "relu4_3"],
        resize=False,
    ):
        super().__init__()

        self.resize = resize
        if vgg_backbone == "VGG16":
            backbone = models.vgg16(
                weights=models.VGG16_Weights.IMAGENET1K_V1
            ).features.eval()
        elif vgg_backbone == "VGG19":
            backbone = models.vgg19(
                weights=models.VGG19_Weights.IMAGENET1K_V1
            ).features.eval()
        else:
            raise ValueError("Only support backbone 'VGG16' and 'VGG19'.")

        with open(vgg16_config_path, "r") as f:
            vgg_config = json.load(f)

        self.content_layers_idx = [vgg_config[layer] for layer in content_layers]
        self.style_layers_idx = [vgg_config[layer] for layer in style_layers]

        max_index = max(self.content_layers_idx + self.style_layers_idx)
        self.backbone = backbone[: max_index + 1]

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def extract_features(self, x):
        # Safety Reason: Clamp the image to the range from [-1, 1]
        x = torch.clamp(x, -1.0, 1.0)

        # Normalize back to the range [0, 1]: VGG model expects that
        # Only shift if the input is likely [-1, 1]
        if x.min() < 0:
            x = (x + 1.0) / 2

        x = (x - self.mean) / self.std
        # Optional but recommended for changing the input to 224x224
        # if the size of the image is less than 224 (deep features might vanishes)
        if self.resize and x.shape[-1] < 224:
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        features = {}

        for name, layer in self.backbone.named_children():
            x = layer(x)
            index = int(name)

            if index in self.content_layers_idx:
                features[f"content_{index}"] = x

            if index in self.style_layers_idx:
                features[f"style_{index}"] = x

        return features

    def gram_matrix(self, x):
        _, c, h, w = x.shape
        gram_matrix = torch.einsum("b c h w, b d h w -> b c d", x, x)
        gram_matrix = gram_matrix / (c * h * w)

        return gram_matrix

    def forward(self, predict, target, content_weight=1.0, style_weight=1e5):
        predict_feat = self.extract_features(predict)
        target_feat = self.extract_features(target)

        # Calculate the content loss (MSE)
        loss_content = 0
        for content_key in predict_feat:
            if content_key.startswith("content"):
                loss_content += F.mse_loss(
                    predict_feat[content_key], target_feat[content_key]
                )

        # Calculate the style loss (MSE)
        loss_style = 0
        for style_key in predict_feat:
            if style_key.startswith("style"):
                predict_gram = self.gram_matrix(predict_feat[style_key])
                target_gram = self.gram_matrix(target_feat[style_key])
                loss_style += F.mse_loss(predict_gram, target_gram)

        return content_weight * loss_content + style_weight * loss_style

