"""CNN model architecture for ball classification."""

import torch
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
            nn.Dropout(0.2),
            BigLittleReduction(15, 12, "relu"),  # Output channels: 18
            SEBlock(18),
            nn.Dropout(0.1),
            BigLittleReduction(18, 12, "relu"),  # Output channels: 18
            SEBlock(18),
            nn.Dropout(0.1),
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
