import argparse
import yaml
from models.ppo.trainer import PPOTrainer
from environments.industrial_arm.arm_env import IndustrialArmEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    env = IndustrialArmEnv(config['env'])
    trainer = PPOTrainer(env, config['training'])
    trainer.train()

if __name__ == "__main__":
    main()
