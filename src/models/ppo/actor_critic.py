import torch
import torch.nn as nn
from torch.distributions import Normal

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_size=256):
        super().__init__()
        
        # Separate encoders for different observation types
        self.qpos_encoder = nn.Linear(7, hidden_size // 4)  # Joint positions
        self.qvel_encoder = nn.Linear(7, hidden_size // 4)  # Joint velocities
        self.pos_encoder = nn.Linear(3, hidden_size // 4)   # End effector position
        self.quat_encoder = nn.Linear(4, hidden_size // 4)  # End effector orientation
        
        self.base = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU()
        )
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, act_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Linear(hidden_size, 1)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def encode_obs(self, obs_dict):
        # Encode each observation component
        qpos_feat = self.qpos_encoder(obs_dict['qpos'])
        qvel_feat = self.qvel_encoder(obs_dict['qvel'])
        pos_feat = self.pos_encoder(obs_dict['eef_pos'])
        quat_feat = self.quat_encoder(obs_dict['eef_quat'])
        
        # Concatenate all features
        return torch.cat([qpos_feat, qvel_feat, pos_feat, quat_feat], dim=-1)

    def forward(self, obs_dict):
        # Encode observations
        features = self.encode_obs(obs_dict)
        # Process through base network
        features = self.base(features)
        # Get action mean and value
        return self.actor(features), self.critic(features)

class PPOTrainer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.clip_eps = config.clip_epsilon
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        
        self.policy = PolicyNetwork(
            config.obs_dim, 
            config.act_dim
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), 
            lr=config.lr, 
            weight_decay=1e-6,
            betas=(0.9, 0.999)
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_timesteps // config.batch_size
        )

    def update(self, rollouts):
        total_loss = 0
        for _ in range(self.num_epochs):
            # Sample mini-batches
            for batch in rollouts.get_batches(self.batch_size):
                obs_dict, actions, returns, advantages, old_log_probs = batch
                
                # Move to device
                obs_dict = {k: v.to(self.device) for k, v in obs_dict.items()}
                actions = actions.to(self.device)
                returns = returns.to(self.device)
                advantages = advantages.to(self.device)
                old_log_probs = old_log_probs.to(self.device)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Get policy outputs
                mu, values = self.policy(obs_dict)
                dist = Normal(mu, torch.exp(self.policy.log_std))
                log_probs = dist.log_prob(actions).sum(-1)
                
                # PPO policy loss
                ratio = torch.exp(log_probs - old_log_probs)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = 0.5 * (returns - values).pow(2).mean()
                
                # Entropy bonus
                entropy_loss = -self.entropy_coef * dist.entropy().mean()
                
                # Total loss
                loss = policy_loss + value_loss + entropy_loss
                
                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
        
        # Update learning rate
        self.scheduler.step()
        
        return {
            'total_loss': total_loss / (self.num_epochs * self.num_batches),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }