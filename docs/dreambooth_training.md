# Dreambooth Training Guide

## Overview

Dreambooth is a fine-tuning technique for stable diffusion models that allows personalization with just a few images. This guide covers the training process using the MLX-optimized implementation.

## Quick Start

```bash
mflux-train-dreambooth \
  --model="schnell" \
  --instance_data_dir="./images" \
  --output_dir="output" \
  --instance_prompt="a photo of xyz" \
  --num_class_images=10
```

## Training Process

### 1. Data Preparation

#### Image Requirements
- Resolution: 512x512 pixels
- Format: RGB images
- Quality: Clear, well-lit images
- Variety: Different angles/poses

#### Directory Structure
```
project/
├── images/           # Instance images
│   ├── img1.jpg
│   └── img2.jpg
└── output/           # Training outputs
```

### 2. Training Configuration

#### Essential Parameters

| Parameter | Description | Best Practice |
|-----------|-------------|---------------|
| model | Base model to fine-tune | "schnell" for fast training |
| instance_data_dir | Directory with training images | Use 5-10 high-quality images |
| output_dir | Where to save results | Regular checkpoints recommended |
| instance_prompt | Text prompt for training | "a photo of [identifier]" |

#### Advanced Parameters

| Parameter | Description | Optimization Tips |
|-----------|-------------|------------------|
| train_batch_size | Images per batch | Keep at 1 for stability |
| gradient_accumulation_steps | Steps before update | 4-8 for effective batching |
| learning_rate | Training rate | 5e-6 for stable training |
| lr_scheduler | LR adjustment method | "cosine" for smooth decay |
| lr_warmup_steps | Gradual LR increase | 50 steps recommended |
| max_train_steps | Total training steps | 800-1000 for good results |
| guidance | Generation guidance scale | 7.5 for balance |

### 3. Training Process

1. **Initialization**
   - Model loading
   - VAE preparation
   - Data pipeline setup

2. **Training Loop**
   - Image preprocessing
   - Forward pass
   - Loss calculation
   - Gradient accumulation
   - Weight updates

3. **Monitoring**
   - Loss tracking
   - Image generation
   - Model checkpoints

### 4. Output Analysis

#### Generated Images
- Check for subject likeness
- Verify prompt adherence
- Assess image quality

#### Loss Metrics
- Training loss should decrease
- Monitor for convergence
- Check for instabilities

## Optimization Tips

### 1. Image Quality

- Use clear, high-resolution images
- Ensure good lighting
- Remove background clutter
- Maintain consistent style

### 2. Training Stability

- Start with low learning rate
- Use gradient accumulation
- Monitor loss curves
- Save regular checkpoints

### 3. Performance

- Use Metal acceleration
- Optimize batch size
- Balance steps vs quality
- Monitor memory usage

## Troubleshooting

### Common Issues

1. **Poor Results**
   - Check image quality
   - Adjust learning rate
   - Increase training steps
   - Refine prompt

2. **Training Crashes**
   - Reduce batch size
   - Check memory usage
   - Verify image format
   - Update dependencies

3. **Slow Training**
   - Enable Metal
   - Optimize parameters
   - Check CPU/GPU usage
   - Reduce image size

## Best Practices

### 1. Data Preparation
- Clean, consistent images
- Proper preprocessing
- Varied perspectives
- Good metadata

### 2. Training Configuration
- Start conservative
- Regular evaluation
- Incremental changes
- Good documentation

### 3. Monitoring
- Track metrics
- Visual inspection
- Regular backups
- Performance logs

## Advanced Usage

### Custom Training

```python
from mflux.dreambooth_training import Trainer

trainer = Trainer(
    model_name="schnell",
    instance_data_dir="./images",
    output_dir="./output",
    instance_prompt="a photo of xyz",
    num_class_images=10
)

trainer.train(
    max_steps=800,
    learning_rate=5e-6,
    guidance_scale=7.5
)
```

### Experiment Tracking

```python
from mflux.utils import MetricsLogger

logger = MetricsLogger()
trainer.train(
    max_steps=800,
    callbacks=[logger.log_metrics]
)
```

## Future Development

### Planned Features
1. Advanced augmentation
2. Adaptive training
3. Multi-GPU support
4. Enhanced monitoring

### Optimization Goals
1. Faster training
2. Better quality
3. Lower memory usage
4. Improved stability
