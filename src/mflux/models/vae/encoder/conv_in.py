import mlx.core as mx
import mlx.nn as nn


class ConvIn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(
            in_channels=3,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

    def forward(self, input_array: mx.array) -> mx.array:
        # MLX attend les données au format NHWC (batch, height, width, channels)
        input_array = mx.transpose(input_array, (0, 2, 3, 1))
        output = self.conv2d(input_array)
        # Retourner au format NCHW (batch, channels, height, width)
        return mx.transpose(output, (0, 3, 1, 2))
