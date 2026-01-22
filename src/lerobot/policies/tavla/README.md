# TA-VLA Policy for LeRobot

TA-VLA (Task-Aware Vision-Language-Action) policy integration for LeRobot, specifically designed for OpenArm robots.

## Overview

This policy extends PI0 with TA-VLA-specific data processing:
- **Camera mapping**: Maps OpenArm camera names (`cam_high`, `cam_left_wrist`, `cam_right_wrist`) to TA-VLA format (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`)
- **Dimension handling**: Pads state/actions from 14 dims (OpenArm) to model's `max_action_dim` (32), and truncates outputs back to 14 dims
- **Effort support**: Handles effort/torque signals from OpenArm motors
- **Model architecture**: Based on PI0, using PaliGemma vision encoder and Gemma action expert

## Key Components

### 1. Configuration (`configuration_tavla.py`)

`TavlaConfig` extends `PreTrainedConfig` with TA-VLA-specific settings:
- `action_dim`: Real action dimension (14 for OpenArm)
- `max_action_dim`: Padded dimension for model (32)
- `use_wrist_cameras`: Whether to use wrist cameras
- `use_effort`: Whether to include effort/torque signals
- `effort_dim`: Effort dimension (14 for OpenArm)

### 2. Data Processing (`processor_tavla.py`)

**TavlaInputProcessor**: Converts LeRobot format to TA-VLA format
- Maps camera names
- Pads state to `max_state_dim`
- Handles effort signals

**TavlaOutputProcessor**: Converts TA-VLA format back to LeRobot format
- Truncates actions from `max_action_dim` to `action_dim`

### 3. Model (`modeling_tavla.py`)

`TavlaPolicy` wraps `PI0Pytorch` with TA-VLA-specific preprocessing:
- Uses PI0 architecture (PaliGemma + Gemma action expert)
- Handles TA-VLA data format conversion
- Supports effort/torque signals

## Usage

### Basic Usage

```python
from lerobot.policies.tavla import TavlaConfig
from lerobot.policies.factory import get_policy_class

# Create config
config = TavlaConfig(
    action_dim=14,  # OpenArm has 14 joints
    max_action_dim=32,
    use_wrist_cameras=True,
    use_effort=True,
    effort_dim=14,
)

# Get policy class
PolicyClass = get_policy_class("tavla")

# Create policy
policy = PolicyClass(config)
```

### Training

```python
from lerobot.policies.factory import make_policy_config

# Create config for training
config = make_policy_config(
    "tavla",
    action_dim=14,
    max_action_dim=32,
    use_wrist_cameras=True,
    use_effort=True,
    # ... other training parameters
)

# Use with lerobot training pipeline
```

### Inference

```python
# Load pretrained model (compatible with PI0 checkpoints)
policy = TavlaPolicy.from_pretrained(
    "path/to/checkpoint",
    config=config,
)

# Get action from observation
observation = {
    "observation.images.cam_high": base_image,  # [B, 3, 224, 224]
    "observation.images.cam_left_wrist": left_wrist_image,  # [B, 3, 224, 224]
    "observation.images.cam_right_wrist": right_wrist_image,  # [B, 3, 224, 224]
    "observation.state": state,  # [B, 14]
    "observation.effort": effort,  # [B, 14] (optional)
}

action = policy.select_action(observation)  # [B, 14]
```

## Data Format

### Input Format (LeRobot)

- `observation.images.cam_high`: [B, 3, H, W] or [B, H, W, 3]
- `observation.images.cam_left_wrist`: [B, 3, H, W] (optional)
- `observation.images.cam_right_wrist`: [B, 3, H, W] (optional)
- `observation.state`: [B, 14] (OpenArm joint angles)
- `observation.effort`: [B, 14] (OpenArm joint torques, optional)

### Internal Format (TA-VLA)

- `image.base_0_rgb`: [B, H, W, 3] (normalized to [-1, 1])
- `image.left_wrist_0_rgb`: [B, H, W, 3] (optional)
- `image.right_wrist_0_rgb`: [B, H, W, 3] (optional)
- `image_mask.*`: [B] (boolean masks)
- `state`: [B, 32] (padded from 14)
- `effort`: [B, 14] (optional)

### Output Format (LeRobot)

- `action`: [B, chunk_size, 14] (truncated from 32)

## Model Architecture

TA-VLA uses the same architecture as PI0:
- **Vision Encoder**: PaliGemma (SigLIP vision tower + Gemma language model)
- **Action Expert**: Gemma-based transformer for action prediction
- **Flow Matching**: For action generation

## Notes

- TA-VLA models are compatible with PI0 checkpoints (same architecture)
- The main difference is in data preprocessing/format conversion
- Effort signals are currently passed through but not fully utilized in the model (can be extended)
- Camera images are normalized to [-1, 1] for SigLIP encoder

## Future Extensions

- Full integration of effort signals into model architecture
- Support for different camera configurations
- Custom loss functions for TA-VLA specific tasks
- Integration with TA-VLA's original training pipeline

