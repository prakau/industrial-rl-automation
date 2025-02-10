import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Dict, List, Tuple

class PPOTrainer:
    def __init__(self, config: Dict):
        self.device = torch.device(config["device"])
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_ratio = config["clip_ratio"]
        
        self.policy = self._build_policy(config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), 
            lr=config["learning_rate"]
        )

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        # Normalize advantages
        advantages = (batch["advantages"] - batch["advantages"].mean()) / \
                    (batch["advantages"].std() + 1e-8)
                    
        # Policy loss
        ratio = torch.exp(self.policy.log_prob(batch["obs"], batch["actions"]) - 
                         batch["old_log_probs"])
        clip_adv = torch.clamp(ratio, 1-self.clip_ratio, 1+self.clip_ratio) * advantages
        policy_loss = -torch.min(ratio * advantages, clip_adv).mean()
        
        # Value loss
        value_loss = ((self.policy.value(batch["obs"]) - batch["returns"]) ** 2).mean()
        
        # Update policy
        loss = policy_loss + 0.5 * value_loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item()
        }

    def _build_policy(self, config: Dict) -> nn.Module:
        return ActorCritic(
            obs_dim=config["obs_dim"],
            action_dim=config["action_dim"],
            hidden_dim=config["hidden_dim"]
        )

    def compute_returns(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        last_gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t + 1] * masks[t] - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * masks[t] * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
            
        return returns, advantages

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(obs)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean, value = self(obs)
        dist = Normal(action_mean, torch.exp(self.actor_logstd))
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().mean()
        return log_prob, value, entropy
