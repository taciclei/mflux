# MLX Variational Autoencoder (VAE)

## Overview

The VAE module is a key component in the Dreambooth training pipeline, optimized for MLX and Metal performance on Apple Silicon. It handles the encoding and decoding of images in the latent space, which is crucial for stable diffusion training.

## Architecture

### Components

1. **ResBlock**
   - Residual blocks with skip connections
   - Dual convolution layers with LeakyReLU activation
   - Helps maintain gradient flow and feature preservation

2. **Encoder**
   - Input: RGB images [B, C, H, W] in NCHW format
   - Progressive downsampling with increasing channel depth
   - Structure:
     ```
     Input (3) -> Conv2d (64) -> Down1 (128) -> Down2 (256) -> Down3 (512)
     -> Bottleneck -> Projection (mu, logvar)
     ```

3. **Decoder**
   - Input: Latent vectors from encoder
   - Progressive upsampling with decreasing channel depth
   - Structure:
     ```
     Input -> Conv2d (512) -> Bottleneck -> Up1 (256) -> Up2 (128) 
     -> Up3 (64) -> Output (3)
     ```

4. **VAE**
   - Combines Encoder and Decoder
   - Handles format conversion (NCHW ↔ NHWC)
   - Implements reparameterization trick
   - Normalizes inputs/outputs

## Usage

### Basic Usage

```python
import mlx.core as mx
from mflux.models.vae import VAE

# Initialize VAE
vae = VAE(latent_channels=4)

# Encode image to latent space
image = mx.array(...)  # [B, C, H, W] format, range [0, 255]
latent = vae.encode(image)

# Decode back to image space
reconstructed = vae.decode(latent)
```

### Training Configuration

```bash
mflux-train-dreambooth \
  --model="schnell" \
  --instance_data_dir="./images" \
  --output_dir="output" \
  --instance_prompt="a photo of xyz" \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=5e-6 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=50 \
  --max_train_steps=800 \
  --guidance=7.5 \
  --num_class_images=10
```

#### Key Parameters

| Parameter | Description | Default | Range |
|-----------|-------------|---------|--------|
| train_batch_size | Number of images per training step | 1 | 1-4 |
| gradient_accumulation_steps | Steps before weight update | 4 | 1-8 |
| learning_rate | Learning rate for training | 5e-6 | 1e-6 to 1e-5 |
| guidance | Classifier-free guidance scale | 7.5 | 5.0-10.0 |
| num_class_images | Number of class images to generate | 10 | 5-100 |

## Implementation Details

### Data Format

The VAE expects and produces images in the following format:
- Input: NCHW format, range [0, 255]
- Internal: NHWC format, range [-1, 1]
- Output: NCHW format, range [0, 255]

### Tensor Shapes

```python
# Example shapes for 512x512 images
input_image: [1, 3, 512, 512]    # NCHW
latent:      [1, 4, 64, 64]      # NCHW, 8x downsampled
output:      [1, 3, 512, 512]    # NCHW
```

### Key Features

1. **Optimized for Metal**
   - Uses NHWC format internally for better performance
   - Leverages MLX's native operations

2. **Stable Training**
   - ResBlocks for gradient stability
   - LeakyReLU to prevent dead neurons
   - Proper initialization and normalization

3. **Logging System**
   - Configurable logging levels
   - Shape tracking for debugging
   - Performance monitoring

## Debugging

### Logging Levels

```python
import logging

# For detailed debug information
logger.setLevel(logging.DEBUG)

# For normal operation
logger.setLevel(logging.INFO)

# For minimal output
logger.setLevel(logging.WARNING)
```

### Common Issues

1. **Memory Issues**
   - Reduce batch size
   - Use gradient accumulation
   - Monitor VRAM usage

2. **Training Instability**
   - Reduce learning rate
   - Increase warmup steps
   - Adjust guidance scale

3. **Poor Image Quality**
   - Check normalization
   - Verify data preprocessing
   - Adjust model capacity

## Performance Optimization

1. **Memory Usage**
   - Batch size of 1 with gradient accumulation
   - Progressive loading of images
   - Efficient tensor management

2. **Speed**
   - NHWC format for Metal
   - Optimized convolution operations
   - Minimal format conversions

3. **Quality**
   - Proper initialization
   - Stable gradient flow
   - Balanced architecture

## Future Improvements

1. **Planned Features**
   - Adaptive learning rate
   - Dynamic batch sizing
   - Advanced logging metrics

2. **Optimizations**
   - Further Metal optimizations
   - Memory usage improvements
   - Training speed enhancements
