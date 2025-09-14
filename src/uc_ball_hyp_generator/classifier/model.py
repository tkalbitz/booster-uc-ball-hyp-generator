"""CNN model architecture for ball classification."""

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from uc_ball_hyp_generator.layer.big_little_reduction import BigLittleReduction
from uc_ball_hyp_generator.layer.conv2dbn import Conv2dBn
from uc_ball_hyp_generator.layer.seblock import SEBlock


class BallClassifier(nn.Module):
    """
    An efficient CNN for binary ball classification, based on a BigLittle-style
    feature extractor.
    """

    def __init__(self, input_size: int = 32) -> None:
        """
        Initializes the ball classifier model.

        Args:
            input_size: The height and width of the square input patches (default: 32).
        """
        super().__init__()

        # Feature extractor using a sequence of custom layers.
        # This structure follows modern best practices by encapsulating
        # the feature extraction logic in a single sequential module.
        self.features = nn.Sequential(
            nn.BatchNorm2d(3),
            Conv2dBn(3, 8, kernel_size=3, padding="same", activation="relu"),
            BigLittleReduction(8, 10, "relu"),  # Output channels: 15
            nn.Dropout(0.3),
            BigLittleReduction(15, 12, "relu"),  # Output channels: 18
            SEBlock(18),
            nn.Dropout(0.2),
            BigLittleReduction(18, 12, "relu"),  # Output channels: 18
            SEBlock(18),
            nn.Dropout(0.2),
            BigLittleReduction(18, 14, "relu"),  # Output channels: 21
            SEBlock(21),
            nn.Dropout(0.1),
        )

        # To make the model robust to input_size changes, we dynamically
        # calculate the number of flattened features after the feature extractor.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, input_size, input_size)
            output_shape = self.features(dummy_input).shape
            flattened_features = output_shape[1] * output_shape[2] * output_shape[3]

        # Classifier head for binary classification.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_features, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass of the classifier.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Output tensor of shape (B, 1) with probabilities.
        """
        x = self.features(x)
        x = self.classifier(x)
        return x


class BallClassifierHypercolumn(nn.Module):
    """
    A hypercolumn variant of the BallClassifier.

    This model collects feature maps from multiple layers, concatenates them,
    and uses this richer "hypercolumn" vector for classification.
    """

    def __init__(self) -> None:
        """Initializes the hypercolumn ball classifier model."""
        super().__init__()

        # Define layers individually to capture intermediate outputs
        self.input_bn = nn.BatchNorm2d(3)
        self.conv_initial = Conv2dBn(3, 8, kernel_size=3, padding="same", activation="relu")

        self.block1 = nn.Sequential(
            BigLittleReduction(8, 10, "relu"),  # Out: 15 channels
            nn.Dropout(0.2),
        )
        self.block2 = nn.Sequential(
            BigLittleReduction(15, 12, "relu"),  # Out: 18 channels
            SEBlock(18),
            nn.Dropout(0.1),
        )
        self.block3 = nn.Sequential(
            BigLittleReduction(18, 12, "relu"),  # Out: 18 channels
            SEBlock(18),
            nn.Dropout(0.1),
        )
        self.block4 = nn.Sequential(
            BigLittleReduction(18, 14, "relu"),  # Out: 21 channels
            SEBlock(21),
            nn.Dropout(0.1),
        )

        # Calculate the total number of channels for the hypercolumn vector
        # by summing the channels from each captured feature map.
        hypercolumn_channels = 8 + 15 + 18 + 18 + 21  # Total: 80

        # Classifier head that operates on the concatenated hypercolumn vector
        self.hypercolumn_classifier = nn.Sequential(
            nn.Linear(hypercolumn_channels, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Defines the forward pass with hypercolumn creation.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Output tensor of shape (B, 1) with probabilities.
        """
        # List to store feature maps from different depths
        feature_maps = []

        # Pass input through layers and capture outputs
        x = self.input_bn(x)

        f0 = self.conv_initial(x)
        feature_maps.append(f0)

        f1 = self.block1(f0)
        feature_maps.append(f1)

        f2 = self.block2(f1)
        feature_maps.append(f2)

        f3 = self.block3(f2)
        feature_maps.append(f3)

        f4 = self.block4(f3)
        feature_maps.append(f4)

        # Pool each feature map to a vector of size (B, C)
        pooled_vectors = [torch.flatten(F.adaptive_avg_pool2d(fm, (1, 1)), 1) for fm in feature_maps]

        # Concatenate all pooled vectors to form the hypercolumn
        hypercolumn = torch.cat(pooled_vectors, dim=1)

        # Pass the rich hypercolumn vector to the classifier
        return self.hypercolumn_classifier(hypercolumn)
