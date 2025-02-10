# Project Title

A short description of your project.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- MuJoCo
- ROS (optional, for real robot deployment)
- CUDA-capable GPU (recommended for training)

### Installation
1. Clone the repository
   ```
   git clone https://github.com/username/repository.git
   ```
2. Navigate into the project directory
   ```
   cd repository
   ```
3. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. If using ROS (for real robot deployment), install ROS dependencies separately:
   ```bash
   # For ROS Noetic (Ubuntu 20.04)
   sudo apt install ros-noetic-ros-base python3-rospy python3-catkin-pkg

   # For ROS2 (optional)
   sudo apt install ros-humble-ros-base python3-rclpy
   ```

## Usage

### Quick Start

1. Setup virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run training:
   ```bash
   python src/train.py --config src/configs/train.yaml
   ```

3. (Optional) Run tests:
   ```bash
   pytest
   ```

### How to Proceed

- Modify configuration files under `src/configs/` to adjust training and environment settings.
- Customize the MuJoCo environment in `src/environments/industrial_arm/arm_env.py` as needed.
- Update or extend RL algorithm implementations in `src/models/ppo/trainer.py`.
- For debugging, use VS Code’s built-in debugger with settings defined in `.vscode/settings.json`.
- Monitor training with Weights & Biases by logging in with:
  ```bash
  wandb login
  ```

## Project Structure

- `src/environments/`: Custom MuJoCo environments
- `src/models/`: RL algorithm implementations
- `src/configs/`: Configuration files
- `src/scripts/`: Deployment and utility scripts
- `notebooks/`: Jupyter notebooks for experiments
- `docs/`: Documentation

## Training

```bash
python src/train.py --config src/configs/train.yaml
```

## Deployment

1. Build ROS workspace:
   ```bash
   cd ros_ws/
   catkin_make
   source devel/setup.bash
   ```

2. Launch the ROS bridge:
   ```bash
   roslaunch industrial_rl_bridge bridge.launch policy_path:=/path/to/policy.pt
   ```

## Development

- Use `black` for code formatting
- Run tests with `pytest`
- Monitor training with Weights & Biases

## Monitoring

Monitor training progress and metrics using Weights & Biases:

## Technical Details

### Environment
- Custom MuJoCo environment with 7-DoF robotic arm
- Domain randomization for sim2real transfer
- Configurable reward shaping and curriculum learning

### Neural Network Architecture
- Actor-Critic architecture with shared backbone
- Policy: Multi-layer perceptron with LayerNorm and GELU
- State-dependent action noise for exploration

### Training Features
- PPO with clipped objective
- GAE-Lambda advantage estimation
- Multi-environment vectorized training
- Curriculum learning support

## Advanced Usage

### Custom Environment

## Configuration

### Environment Parameters

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
