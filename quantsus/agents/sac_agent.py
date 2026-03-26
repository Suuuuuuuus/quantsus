import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .networks import Actor, Critic
from .replay_buffer import ReplayBuffer

LEARN_ALPHA_RATE = 3e-4
DECAY_GAMMA = 0.99
NETWORK_UPDATE_TAU = 0.005

class SACAgent:
    def __init__(self, state_dim, action_dim,
                 learn_rate = LEARN_ALPHA_RATE, device="cpu", 
                 decay_gamma = DECAY_GAMMA, network_update_tau = NETWORK_UPDATE_TAU):
        
        self.device = torch.device(device)
        self.state_dim = state_dim
        self.action_dim = action_dim

        # networks
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic1 = Critic(state_dim, action_dim).to(self.device)
        self.target_critic2 = Critic(state_dim, action_dim).to(self.device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        # optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=learn_rate)
        self.critic1_opt = optim.Adam(self.critic1.parameters(), lr=learn_rate)
        self.critic2_opt = optim.Adam(self.critic2.parameters(), lr=learn_rate)

        # entropy tuning
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=learn_rate)
        self.target_entropy = -action_dim
        
        self.buffer = ReplayBuffer(state_dim, action_dim)

        # hyperparams
        self.gamma = decay_gamma
        self.tau = network_update_tau

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if deterministic:
            mu, _ = self.actor(state)
            action = torch.tanh(mu)
        else:
            action, _ = self.actor.sample(state)

        return action.detach().cpu().numpy()[0]

    def update(self, batch_size=256):
        if self.buffer.size < batch_size:
            return

        batch = self.buffer.sample(batch_size)
        s = batch["state"].to(self.device)
        a = batch["action"].to(self.device)
        r = batch["reward"].to(self.device)
        s2 = batch["next_state"].to(self.device)
        d = batch["done"].to(self.device)

        # -------- critic update --------
        with torch.no_grad():
            a2, logp_a2 = self.actor.sample(s2)

            q1_target = self.target_critic1(s2, a2)
            q2_target = self.target_critic2(s2, a2)
            q_target = torch.min(q1_target, q2_target) - self.alpha * logp_a2

            backup = r + self.gamma * (1 - d) * q_target

        q1 = self.critic1(s, a)
        q2 = self.critic2(s, a)

        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        self.critic1_opt.zero_grad()
        loss_q1.backward()
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        loss_q2.backward()
        self.critic2_opt.step()

        # -------- actor update --------
        a_pi, logp_a_pi = self.actor.sample(s)
        q1_pi = self.critic1(s, a_pi)
        q2_pi = self.critic2(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)

        actor_loss = (self.alpha * logp_a_pi - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # -------- alpha update --------
        alpha_loss = -(self.log_alpha * (logp_a_pi + self.target_entropy).detach()).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # -------- target update --------
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



def save_sac(agent, path):
    """
    Save SAC agent weights and optimizer states.
    """
    checkpoint = {
        "state_dim": agent.state_dim,
        "action_dim": agent.action_dim,
        "actor_state_dict": agent.actor.state_dict(),
        "critic1_state_dict": agent.critic1.state_dict(),
        "critic2_state_dict": agent.critic2.state_dict(),
        "target_critic1_state_dict": agent.target_critic1.state_dict(),
        "target_critic2_state_dict": agent.target_critic2.state_dict(),
        "actor_opt_state_dict": agent.actor_opt.state_dict(),
        "critic1_opt_state_dict": agent.critic1_opt.state_dict(),
        "critic2_opt_state_dict": agent.critic2_opt.state_dict(),
        "log_alpha": agent.log_alpha,
        "alpha_opt_state_dict": agent.alpha_opt.state_dict(),
        "gamma": agent.gamma,
        "tau": agent.tau,
    }
    
    torch.save(checkpoint, path)
    print(f"SAC agent saved to {path}")
    return None

def load_sac(path, device="cpu"):
    """
    Load a saved SAC checkpoint into an existing agent.
    """
    checkpoint = torch.load(path, map_location=device)
    agent = SACAgent(state_dim=checkpoint["state_dim"], action_dim=checkpoint["action_dim"], device=device)
    
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic1.load_state_dict(checkpoint["critic1_state_dict"])
    agent.critic2.load_state_dict(checkpoint["critic2_state_dict"])
    agent.target_critic1.load_state_dict(checkpoint["target_critic1_state_dict"])
    agent.target_critic2.load_state_dict(checkpoint["target_critic2_state_dict"])
    
    agent.actor_opt.load_state_dict(checkpoint["actor_opt_state_dict"])
    agent.critic1_opt.load_state_dict(checkpoint["critic1_opt_state_dict"])
    agent.critic2_opt.load_state_dict(checkpoint["critic2_opt_state_dict"])
    
    agent.log_alpha = checkpoint["log_alpha"]
    agent.alpha_opt.load_state_dict(checkpoint["alpha_opt_state_dict"])
    
    agent.gamma = checkpoint["gamma"]
    agent.tau = checkpoint["tau"]
    
    print(f"SAC agent loaded from {path}")
    return agent