"""
DeepLabV3+ implementation for building damage assessment.

Based on Chen et al. (2018) "Encoder-Decoder with Atrous Separable Convolution
for Semantic Image Segmentation"

Uses ResNet50 backbone with Atrous Spatial Pyramid Pooling (ASPP) module
and a decoder with low-level feature fusion.

Input: 6 channels (pre-disaster RGB + post-disaster RGB concatenated)
Output: 5 channels (logits for damage classes: background, no-damage, minor, major, destroyed)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module.

    Applies parallel atrous convolutions with different rates to capture
    multi-scale contextual information.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels per branch
        atrous_rates (list): List of dilation rates for atrous convolutions
    """

    def __init__(self, in_channels, out_channels=256, atrous_rates=[6, 12, 18]):
        super(ASPP, self).__init__()

        # 1x1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Atrous convolutions with different rates
        self.atrous_blocks = nn.ModuleList()
        for rate in atrous_rates:
            self.atrous_blocks.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3,
                             padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        # Global average pooling branch
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Projection layer to combine all branches
        # Total channels: out_channels * (1 + len(atrous_rates) + 1)
        total_channels = out_channels * (len(atrous_rates) + 2)
        self.project = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        """
        Forward pass through ASPP.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (N, out_channels, H, W)
        """
        size = x.shape[2:]

        # 1x1 convolution
        feat1 = self.conv1x1(x)

        # Atrous convolutions
        atrous_feats = [block(x) for block in self.atrous_blocks]

        # Global pooling
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=size, mode='bilinear', align_corners=False)

        # Concatenate all features
        features = [feat1] + atrous_feats + [global_feat]
        out = torch.cat(features, dim=1)

        # Project to desired output channels
        out = self.project(out)

        return out


class DeepLabV3Plus(nn.Module):
    """
    DeepLabV3+ architecture with ResNet50 backbone.

    Args:
        in_channels (int): Number of input channels (default: 6 for pre+post RGB)
        num_classes (int): Number of output classes (default: 5 for damage levels)
        pretrained (bool): Whether to use ImageNet pretrained ResNet50 weights (default: True)
        output_stride (int): Output stride of the backbone (8 or 16, default: 16)
    """

    def __init__(self, in_channels=6, num_classes=5, pretrained=True, output_stride=16):
        super(DeepLabV3Plus, self).__init__()

        # Load pretrained ResNet50
        resnet = models.resnet50(pretrained=pretrained)

        # Modify first conv layer to accept 6 channels
        if in_channels != 3:
            original_first_conv = resnet.conv1
            new_first_conv = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

            if pretrained:
                # Initialize new conv layer weights by duplicating pretrained weights
                with torch.no_grad():
                    weight = original_first_conv.weight.clone()
                    new_first_conv.weight[:, :3] = weight
                    new_first_conv.weight[:, 3:] = weight  # Duplicate for post-disaster channels

            resnet.conv1 = new_first_conv

        # Encoder: ResNet50 layers
        self.conv1 = resnet.conv1  # stride 2
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # stride 2

        self.layer1 = resnet.layer1  # stride 1 (total stride 4)
        self.layer2 = resnet.layer2  # stride 2 (total stride 8)
        self.layer3 = resnet.layer3  # stride 2 (total stride 16)
        self.layer4 = resnet.layer4  # stride 2 (total stride 32)

        # Modify layer4 to use atrous convolutions if output_stride=16
        if output_stride == 16:
            # Replace stride with dilation in layer4
            self._replace_stride_with_dilation(self.layer4, dilation=2)

        # ASPP module (applied after layer4)
        # layer4 output channels: 2048 for ResNet50
        self.aspp = ASPP(in_channels=2048, out_channels=256, atrous_rates=[6, 12, 18])

        # Low-level feature projection (from layer1 output)
        # layer1 output channels: 256 for ResNet50
        self.low_level_reduce = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Final classifier
        self.classifier = nn.Conv2d(256, num_classes, kernel_size=1)

        # Initialize decoder and classifier weights
        self._initialize_weights()

    def _replace_stride_with_dilation(self, layer, dilation):
        """Replace stride with dilation in a layer to maintain resolution."""
        # Only modify the first block's downsampling
        # This is the standard approach in DeepLabV3+
        for i, block in enumerate(layer):
            if hasattr(block, 'conv2'):
                # Replace stride in conv2 of the block
                conv2 = block.conv2
                if conv2.stride == (2, 2):
                    conv2.stride = (1, 1)
                    conv2.dilation = (dilation, dilation)
                    conv2.padding = (dilation, dilation)
                elif conv2.dilation == (1, 1) and conv2.kernel_size == (3, 3):
                    conv2.dilation = (dilation, dilation)
                    conv2.padding = (dilation, dilation)

            # Handle downsampling layer
            if hasattr(block, 'downsample') and block.downsample is not None:
                for module in block.downsample.modules():
                    if isinstance(module, nn.Conv2d):
                        if module.stride == (2, 2):
                            module.stride = (1, 1)

    def _initialize_weights(self):
        """Initialize decoder and classifier weights."""
        for m in [self.low_level_reduce, self.decoder, self.classifier]:
            for module in m.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.constant_(module.weight, 1)
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        """
        Forward pass through DeepLabV3+.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 6, H, W)

        Returns:
            torch.Tensor: Output logits of shape (N, num_classes, H, W)
        """
        input_size = x.shape[2:]

        # Encoder path
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Low-level features (stride 4)
        low_level_feat = self.layer1(x)

        # High-level features
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)  # stride 16 (with atrous convolutions)

        # ASPP
        x = self.aspp(x)

        # Upsample to match low-level features (4x upsampling)
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=False)

        # Reduce low-level feature channels
        low_level_feat = self.low_level_reduce(low_level_feat)

        # Concatenate high-level and low-level features
        x = torch.cat([x, low_level_feat], dim=1)

        # Decoder
        x = self.decoder(x)

        # Classifier
        x = self.classifier(x)

        # Upsample to input resolution (4x upsampling)
        x = F.interpolate(x, size=input_size, mode='bilinear', align_corners=False)

        return x


def test_deeplabv3plus():
    """Test DeepLabV3+ model with sample input."""
    print("Testing DeepLabV3+ model...")

    # Create model
    model = DeepLabV3Plus(in_channels=6, num_classes=5, pretrained=False)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass with sample input
    batch_size = 2
    height, width = 256, 256
    x = torch.randn(batch_size, 6, height, width)

    print(f"Input shape: {x.shape}")

    with torch.no_grad():
        output = model(x)

    print(f"Output shape: {output.shape}")
    print(f"Expected shape: ({batch_size}, 5, {height}, {width})")

    # Verify output shape
    assert output.shape == (batch_size, 5, height, width), "Output shape mismatch!"

    print("✓ DeepLabV3+ test passed!")

    # Test gradient computation
    print("\nTesting gradient computation...")
    x.requires_grad = True
    output = model(x)
    loss = output.sum()
    loss.backward()
    print("✓ Gradient computation successful!")


if __name__ == "__main__":
    test_deeplabv3plus()
