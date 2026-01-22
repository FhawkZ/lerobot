# TA-VLA05 Policy for LeRobot

TA-VLA (Task-Aware Vision-Language-Action) policy integration for LeRobot based on **PI0.5**, specifically designed for OpenArm robots.

## Overview

This policy extends **PI0.5** with TA-VLA-specific data processing:
- **Camera mapping**: Maps OpenArm camera names (`cam_high`, `cam_left_wrist`, `cam_right_wrist`) to TA-VLA format (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`)
- **Dimension handling**: Pads state/actions from 14 dims (OpenArm) to model's `max_action_dim` (32), and truncates outputs back to 14 dims
- **Effort support**: Handles effort/torque signals from OpenArm motors
- **Model architecture**: Based on **PI0.5**, using PaliGemma vision encoder and Gemma action expert with **AdaRMS** and **time_mlp** for time conditioning

## Key Differences from PI0-based TA-VLA

| Feature | TA-VLA (PI0) | TA-VLA05 (PI0.5) |
|---------|--------------|------------------|
| **Base Model** | PI0 | PI0.5 |
| **Time Conditioning** | Concatenates time with actions (`action_time_mlp_*`) | Uses `time_mlp_*` for AdaRMS conditioning |
| **AdaRMS** | Not used | Used in action expert |
| **Normalization** | MEAN_STD | QUANTILES |
| **Tokenizer Length** | 48 tokens | 200 tokens |
| **State Processing** | Uses `state_proj` (PI0 has it) | Uses `state_proj` (added, PI0.5 doesn't have it by default) |
| **Cross-Robot Generalization** | Good | Better (Open-World Generalization) |

## Key Components

### 1. Configuration (`configuration_tavla05.py`)

`Tavla05Config` extends `PreTrainedConfig` with TA-VLA-specific settings:
- `action_dim`: Real action dimension (14 for OpenArm)
- `max_action_dim`: Padded dimension for model (32)
- `use_wrist_cameras`: Whether to use wrist cameras
- `use_effort`: Whether to include effort/torque signals
- `effort_dim`: Effort dimension (14 for OpenArm)
- `normalization_mapping`: Uses QUANTILES (PI0.5 style)
- `tokenizer_max_length`: 200 (PI0.5 style)

### 2. Data Processing (`processor_tavla05.py`)

**Tavla05InputProcessor**: Converts LeRobot format to TA-VLA format
- Maps camera names
- Pads state to `max_state_dim`
- Handles effort signals

**Tavla05OutputProcessor**: Converts TA-VLA format back to LeRobot format
- Truncates actions from `max_action_dim` to `action_dim`

### 3. Model (`modeling_tavla.py`)

`TavlaPolicyPI05` wraps `PI05Pytorch` with TA-VLA-specific preprocessing:
- Uses PI0.5 architecture (PaliGemma + Gemma action expert with AdaRMS)
- Handles TA-VLA data format conversion
- Supports effort/torque signals
- Adds `state_proj` layer (PI0.5 doesn't have it by default, but TA-VLA needs it)
- Uses PI0.5's `time_mlp` and AdaRMS for time conditioning

## Usage

### Basic Usage

```python
from lerobot.policies.tavla05 import Tavla05Config
from lerobot.policies.factory import get_policy_class

# Create config
config = Tavla05Config(
    action_dim=14,
    max_action_dim=32,
    use_effort=True,
    effort_type="expert",
)

# Get policy class
PolicyClass = get_policy_class("tavla05")

# Create policy
policy = PolicyClass(config)
```

### Training

```python
# Training loop
for batch in dataloader:
    loss_dict = policy.compute_loss(batch)
    loss = loss_dict["loss"].mean()
    loss.backward()
    optimizer.step()
```

### Inference

```python
# Get action from observation
observation = {
    "observation.images.cam_high": image_tensor,
    "observation.state": state_tensor,
    "observation.effort": effort_tensor,  # optional
}
action = policy.select_action(observation)
```

## Advantages of PI0.5 Base

1. **Better Cross-Robot Generalization**: QUANTILES normalization makes it easier to transfer between different robot platforms
2. **More Stable Training**: AdaRMS provides adaptive normalization, leading to more stable training
3. **Better Language Understanding**: 200-token limit allows for more complex language instructions
4. **Efficient Time Conditioning**: `time_mlp` + AdaRMS is more efficient than concatenation-based time conditioning

## When to Use TA-VLA05 vs TA-VLA

- **Use TA-VLA05** if:
  - You need better cross-robot generalization
  - You want more stable training
  - You need to handle longer language instructions
  - You're working with multiple robot platforms

- **Use TA-VLA** (PI0-based) if:
  - You're working with a single robot platform
  - You have simpler tasks
  - You want to use existing PI0 pretrained weights directly

## Effort/Torque Support

TA-VLA05 supports multiple effort processing modes:
- `"no"`: No effort signals
- `"expert"`: Pass effort to action expert (default)
- `"llm"`: Pass effort to LLM (PaliGemma)
- `"expert_fut"`: Predict future effort along with actions
- `"expert_his_c"`: Use history of effort, pass to expert
- And more...

See `configuration_tavla05.py` for all available options.

## Loading Pretrained Weights

TA-VLA05 models are compatible with PI0.5 checkpoints:

```python
policy = TavlaPolicyPI05.from_pretrained(
    "path/to/pi05/checkpoint",
    config=config,
)
```

Note: The `state_proj` layer will be randomly initialized since PI0.5 doesn't have it by default.

