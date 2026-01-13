"""
FCN-8s (Fully Convolutional Network) implementation for building damage assessment.

Based on Long et al. (2015) "Fully Convolutional Networks for Semantic Segmentation"
Uses VGG16 backbone with skip connections from pool3, pool4, and final layer.

Input: 6 channels (pre-disaster RGB + post-disaster RGB concatenated)
Output: 5 channels (logits for damage classes: background, no-damage, minor, major, destroyed)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class FCN8s(nn.Module):
    """
    FCN-8s architecture with VGG16 backbone.

    Args:
        in_channels (int): Number of input channels (default: 6 for pre+post RGB)
        num_classes (int): Number of output classes (default: 5 for damage levels)
        pretrained (bool): Whether to use ImageNet pretrained VGG16 weights (default: True)
    """

    def __init__(self, in_channels=6, num_classes=5, pretrained=True):
        super(FCN8s, self).__init__()

        # Load pretrained VGG16
        vgg16 = models.vgg16(pretrained=pretrained)
        features = list(vgg16.features.children())

        # Modify first conv layer to accept 6 channels instead of 3
        if in_channels != 3:
            # Get original first conv layer weights
            original_first_conv = features[0]
            # Create new conv layer with desired input channels
            new_first_conv = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1
            )

            if pretrained:
                # Initialize new conv layer weights
                # Replicate pretrained weights across channels
                with torch.no_grad():
                    # Average the pretrained weights and replicate
                    weight = original_first_conv.weight.clone()
                    new_first_conv.weight[:, :3] = weight
                    new_first_conv.weight[:, 3:] = weight  # Duplicate for post-disaster channels
                    new_first_conv.bias = original_first_conv.bias

            features[0] = new_first_conv

        # VGG16 feature extraction layers
        # Pool3: features[0:17] -> output stride 8x
        # Pool4: features[0:24] -> output stride 16x
        # Pool5: features[0:31] -> output stride 32x

        self.pool3 = nn.Sequential(*features[:17])  # Up to pool3 (1/8 resolution)
        self.pool4 = nn.Sequential(*features[17:24])  # pool3 to pool4 (1/16 resolution)
        self.pool5 = nn.Sequential(*features[24:])  # pool4 to pool5 (1/32 resolution)

        # Classifier layers (replacing VGG16's fully connected layers)
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        # Score layers for each path
        self.score_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)  # 1x1 conv for pool3
        self.score_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)  # 1x1 conv for pool4
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1)  # 1x1 conv for final

        # Upsampling layers (transposed convolutions)
        self.upscore2 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False
        )
        self.upscore8 = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False
        )

        # Initialize upsampling layers with bilinear interpolation weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize transposed convolution weights with bilinear interpolation."""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                # Bilinear interpolation initialization
                c1, c2, h, w = m.weight.data.size()
                weight = self._get_upsampling_weight(c1, c2, h)
                m.weight.data.copy_(weight)
            elif isinstance(m, nn.Conv2d):
                if m not in [self.score_pool3, self.score_pool4, self.score_fr]:
                    continue  # Skip initialization for score layers (keep pretrained or random)
                # Xavier initialization for score layers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _get_upsampling_weight(self, in_channels, out_channels, kernel_size):
        """Generate bilinear interpolation weights for upsampling."""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5

        og = torch.arange(kernel_size).float()
        filt = (1 - torch.abs(og - center) / factor)

        weight = torch.zeros(in_channels, out_channels, kernel_size, kernel_size)
        weight[:, :, :, :] = filt.unsqueeze(0).unsqueeze(0) * filt.unsqueeze(1).unsqueeze(0)

        return weight

    def forward(self, x):
        """
        Forward pass through FCN-8s.

        Args:
            x (torch.Tensor): Input tensor of shape (N, 6, H, W)

        Returns:
            torch.Tensor: Output logits of shape (N, num_classes, H, W)
        """
        # Encoder path (VGG16 backbone)
        pool3_out = self.pool3(x)  # 1/8 resolution
        pool4_out = self.pool4(pool3_out)  # 1/16 resolution
        pool5_out = self.pool5(pool4_out)  # 1/32 resolution

        # Classifier
        x = self.classifier(pool5_out)

        # Score final layer
        score_fr = self.score_fr(x)  # (N, num_classes, H/32, W/32)

        # Upsample 2x and add skip connection from pool4
        upscore2 = self.upscore2(score_fr)  # (N, num_classes, H/16, W/16)
        score_pool4 = self.score_pool4(pool4_out)  # (N, num_classes, H/16, W/16)

        # Add skip connection
        fuse_pool4 = upscore2 + score_pool4  # Element-wise addition

        # Upsample 2x and add skip connection from pool3
        upscore_pool4 = self.upscore_pool4(fuse_pool4)  # (N, num_classes, H/8, W/8)
        score_pool3 = self.score_pool3(pool3_out)  # (N, num_classes, H/8, W/8)

        # Add skip connection
        fuse_pool3 = upscore_pool4 + score_pool3  # Element-wise addition

        # Final 8x upsampling to original resolution
        out = self.upscore8(fuse_pool3)  # (N, num_classes, H, W)

        return out


def test_fcn8s():
    """Test FCN-8s model with sample input."""
    print("Testing FCN-8s model...")

    # Create model
    model = FCN8s(in_channels=6, num_classes=5, pretrained=False)

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

    print("✓ FCN-8s test passed!")

    # Test gradient computation
    print("\nTesting gradient computation...")
    x.requires_grad = True
    output = model(x)
    loss = output.sum()
    loss.backward()
    print("✓ Gradient computation successful!")


if __name__ == "__main__":
    test_fcn8s()
