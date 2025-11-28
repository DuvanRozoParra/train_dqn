# DQNAgent.py
import os
import json
import pickle
import math
import random
from collections import deque, namedtuple
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ------------------------
# Util: device
# ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------
# Noisy Linear (Factorized Gaussian) - Fortemente recomendado para exploration sin epsilon
# ------------------------
class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        # Factorized noise
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.ger(eps_in))
        self.bias_epsilon.copy_(eps_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


# ------------------------
# Simple MLP DQN; optionally replace some linear layers with NoisyLinear
# ------------------------
class DQN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layers: List[int],
        output_size: int,
        hidden_activation: str = "relu",
        use_noisy: bool = False,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_size

        act_fn = nn.ReLU if hidden_activation == "relu" else nn.Tanh

        for i, h in enumerate(hidden_layers):
            if use_noisy:
                layers.append(NoisyLinear(prev, h))
            else:
                layers.append(nn.Linear(prev, h))
            layers.append(act_fn())
            prev = h

        # Output layer
        if use_noisy:
            layers.append(NoisyLinear(prev, output_size))
        else:
            layers.append(nn.Linear(prev, output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


# ------------------------
# Prioritized Replay Buffer (simpler alternative to SumTree)
# - Stores transitions and priorities
# - Sample with probabilities proportional to priority^alpha
# - Update priorities by index
# ------------------------
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta0: float = 0.4, beta_increment_per_sampling: float = 1e-5):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta = beta0
        self.beta_increment = beta_increment_per_sampling

        self.buffer: List[Optional[Transition]] = []
        self.priorities: List[float] = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition: Transition, priority: float = 1.0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = transition
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")

        probs = np.array(self.priorities, dtype=np.float64)
        probs = probs ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        # importance-sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = Transition(*zip(*samples))
        return batch, indices, weights

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        for idx, p in zip(indices, priorities):
            self.priorities[idx] = float(abs(p) + 1e-6)

    def save(self):
        return {"buffer": list(self.buffer), "priorities": list(self.priorities), "pos": self.pos}

    def load(self, data):
        self.buffer = data["buffer"]
        self.priorities = data["priorities"]
        self.pos = data["pos"]


# ------------------------
# Helpers: error functions map and optimizers map
# ------------------------
errores = {
    "mse": nn.MSELoss,
    "huber": lambda: nn.SmoothL1Loss()
}

optimizadores = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "rmsprop": optim.RMSprop
}


# ------------------------
# DQNAgent ULTRA (double, PER optional, NoisyNet optional, soft target updates)
# ------------------------
class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_layers: List[int] = [64, 64],
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99995,
        epsilon_min: float = 0.05,
        batch_size: int = 64,
        memory_size: int = 100_000,
        update_target_every: int = 1000,
        tau: float = 0.0,  # if >0 -> soft updates
        hidden_activation: str = "relu",
        loss_function: str = "huber",
        optimizer: str = "adam",
        use_prioritizedRB: bool = False,
        use_noisy: bool = False,
        use_softmax_action: bool = False,
        softmax_tau: float = 1.0,
        gradient_clip: Optional[float] = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.device = device if device is not None else DEVICE
        #self.device = torch.device("cuda")

        # hyperparams
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.tau = tau

        # exploration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.use_noisy = use_noisy
        self.use_softmax_action = use_softmax_action
        self.softmax_tau = softmax_tau

        self.use_prioritizedRB = use_prioritizedRB
        self.memory_size = memory_size
        self.steps = 0
        self.gradient_clip = gradient_clip

        # Replay
        if use_prioritizedRB:
            self.memory = PrioritizedReplayBuffer(memory_size)
        else:
            self.memory = deque(maxlen=memory_size)

        # Networks
        self.policy_net = DQN(state_size, hidden_layers, action_size, hidden_activation, use_noisy=use_noisy).to(self.device)
        self.target_net = DQN(state_size, hidden_layers, action_size, hidden_activation, use_noisy=use_noisy).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # optimizer & loss
        self.optimizer = optimizadores[optimizer](self.policy_net.parameters(), lr=lr)
        self.criterion = errores[loss_function]()

        # misc
        self.training = True  # control training/inference modes

    # -------------------------
    # Noisy helpers
    # -------------------------
    def reset_noisy(self):
        if hasattr(self.policy_net, "reset_noise"):
            self.policy_net.reset_noise()
        if hasattr(self.target_net, "reset_noise"):
            self.target_net.reset_noise()

    def disable_noisy(self):
        """Disable NoisyNet: set use_noisy False and set epsilon to 0 for deterministic inference."""
        self.use_noisy = False
        self.epsilon = 0.0
        # replace reset_noise with no-op to avoid calls
        for m in self.policy_net.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise = lambda: None
        for m in self.target_net.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise = lambda: None
        print("🔇 Noisy Networks desactivado.")
        
    def enable_noisy(self):
        """Enable NoisyNet exploration (restore reset_noise behavior)."""
        self.use_noisy = True
        # restore reset_noise to actual methods (they exist in NoisyLinear)
        # no need to modify epsilon; NoisyNet ignores epsilon exploration by design
        # ensure networks are in train mode so NoisyLinear uses epsilon buffers
        self.set_train()
        # reset buffers once to initialize epsilons on device
        self.reset_noisy()
        print("🔊 Noisy Networks activado.")
        
    def debug_noisy(self):
        """Print simple stats about NoisyLinear params to verify noise changes."""
        for name, module in self.policy_net.named_modules():
            if isinstance(module, NoisyLinear):
                # .weight_epsilon is a buffer that should change after reset_noise()
                we = module.weight_epsilon.detach().cpu()
                be = module.bias_epsilon.detach().cpu()
                print(f"[NoisyDebug] {name}: weight_epsilon mean {we.mean().item():.6f}, "
                      f"std {we.std().item():.6f} | bias_epsilon mean {be.mean().item():.6f}, std {be.std().item():.6f}")

    # -------------------------
    # Action selection
    # -------------------------
    def select_action(self, state: np.ndarray) -> int:
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Reset noisy layers each step if using NoisyNet
        if self.use_noisy:
            self.reset_noisy()

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)  # shape: (action_size,)

        # --- DEBUG NoisyNet ---
        if self.use_noisy:
            print(f"[Noisy Q-values] {q_values.cpu().numpy()}")

        if self.use_softmax_action:
            probs = torch.softmax(q_values / self.softmax_tau, dim=0).cpu().numpy()
            return int(np.random.choice(self.action_size, p=probs))

        if not self.use_noisy and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        return int(q_values.argmax().item())

    def select_action_greedy(self, state: np.ndarray) -> int:
        """Deterministic action: uses model in eval mode, no epsilon, no noise reset."""
        self.policy_net.eval()
        if isinstance(state, np.ndarray):
            state_tensor = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze(0)
        self.policy_net.train()
        return int(q_values.argmax().item())

    def select_actions_batch(self, states: np.ndarray) -> List[int]:
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)
        if self.use_noisy:
            self.reset_noisy()
        with torch.no_grad():
            q = self.policy_net(states_t)
        actions = q.argmax(dim=1).cpu().numpy()
        return actions.tolist()

    # -------------------------
    # Memory
    # -------------------------
    def store_transition(self, state, action, reward, next_state, done):
        trans = Transition(state, action, reward, next_state, done)
        if self.use_prioritizedRB:
            # priority initial = max priority or 1
            if len(self.memory) == 0:
                p = 1.0
            else:
                p = max(self.memory.priorities) if len(self.memory.priorities) > 0 else 1.0
            self.memory.add(trans, priority=p)
        else:
            self.memory.append(trans)

    # -------------------------
    # Training step (Double DQN)
    # -------------------------
    def train_step(self):
        # Enough samples?
        if self.use_prioritizedRB:
            if len(self.memory) < self.batch_size:
                return
            batch, indices, weights = self.memory.sample(self.batch_size)
            # weights -> numpy -> tensor
            weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            if len(self.memory) < self.batch_size:
                return
            batch_samples = random.sample(self.memory, self.batch_size)
            batch = Transition(*zip(*batch_samples))
            indices = None
            weights_t = None

        # Convert to tensors (and push to device) efficiently
        states = torch.from_numpy(np.asarray(batch.state)).float().to(self.device)
        actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_states = torch.from_numpy(
            np.asarray(batch.next_state, dtype=np.float32)
        ).float().to(self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(self.device)

        if self.use_prioritizedRB:
            weights_t = torch.tensor(weights, dtype=torch.float32).to(self.device)

        # print(" ---- DEBUG BATCH ----")
        # print("states:", states.shape)
        # print("actions:", actions.shape)
        # print("next_states:", next_states.shape)
        # print("rewards:", rewards.shape)
        # print("dones:", dones.shape)
        # print("-----------------------")


        # Current Q-values for taken actions
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)  # shape: (batch,)

        # Double DQN target: use policy_net to pick next action, but target_net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)  # (batch,1)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)  # (batch,)
            target_q = rewards + (1.0 - dones) * (self.gamma * next_q_values)

        # distribución de acciones de la política
        action_counts = next_actions.squeeze().cpu().numpy()
        unique, counts = np.unique(action_counts, return_counts=True)
        dist = dict(zip(unique, counts))

        print(f"[ActionDist] {dist}")

        # TD error
        td_errors = target_q - q_values

        # Loss (optionally weighted by importance-sampling weights)
        if self.use_prioritizedRB:
            loss_per_sample = F.smooth_l1_loss(q_values, target_q, reduction="none")
            loss = (loss_per_sample * weights_t).mean()
        else:
            loss = self.criterion(q_values, target_q)

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()

        # Optional gradient clipping
        if self.gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.gradient_clip)

        self.optimizer.step()

        # Update priorities if PER
        if self.use_prioritizedRB and indices is not None:
            new_priorities = td_errors.detach().abs().cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, new_priorities)


        # Update epsilon if not using noisy
        if not self.use_noisy:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # target network update (soft or hard)
        self.steps += 1
        if self.tau > 0:
            # soft update
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + policy_param.data * self.tau)
        elif self.steps % self.update_target_every == 0:
            # hard update
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # Small debug print (optional) - you can remove/adjust
        ## print(f"Train loss {loss.item():.6f} eps {self.epsilon:.4f}")
        
        with torch.no_grad():
            q_mean = q_values.mean().item()
            q_std = q_values.std().item()

        print(f"[Q-Stats] mean={q_mean:.3f} | std={q_std:.3f}")

    # -------------------------
    # Save / Load
    # -------------------------
    def save(self, folder: str = "models", name: str = "dqn_agent", save_replay_buffer: bool = True):
        os.makedirs(folder, exist_ok=True)
        p_policy = os.path.join(folder, f"{name}_policy.pt")
        p_target = os.path.join(folder, f"{name}_target.pt")
        p_opt = os.path.join(folder, f"{name}_optim.pt")
        p_meta = os.path.join(folder, f"{name}_meta.json")
        p_replay = os.path.join(folder, f"{name}_replay.pkl")

        torch.save(self.policy_net.state_dict(), p_policy)
        torch.save(self.target_net.state_dict(), p_target)
        torch.save(self.optimizer.state_dict(), p_opt)

        meta = {
            "epsilon": self.epsilon,
            "steps": self.steps,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "memory_size": self.memory_size,
            "use_prioritizedRB": self.use_prioritizedRB,
            "use_noisy": self.use_noisy,
            "device": str(self.device)
        }
        with open(p_meta, "w") as f:
            json.dump(meta, f, indent=4)

        if save_replay_buffer:
            if self.use_prioritizedRB:
                with open(p_replay, "wb") as f:
                    pickle.dump(self.memory.save(), f)
            else:
                with open(p_replay, "wb") as f:
                    pickle.dump(list(self.memory), f)

        return meta

    def load(self, folder: str = "models", name: str = "dqn_agent", load_replay_buffer: bool = True):
        p_policy = os.path.join(folder, f"{name}_policy.pt")
        p_target = os.path.join(folder, f"{name}_target.pt")
        p_opt = os.path.join(folder, f"{name}_optim.pt")
        p_meta = os.path.join(folder, f"{name}_meta.json")
        p_replay = os.path.join(folder, f"{name}_replay.pkl")

        if not (os.path.exists(p_policy) and os.path.exists(p_target) and os.path.exists(p_meta)):
            print("⚠️ No se encontraron pesos/metadata completos.")
            return None

        map_loc = self.device if isinstance(self.device, torch.device) else DEVICE
        self.policy_net.load_state_dict(torch.load(p_policy, map_location=map_loc))
        self.target_net.load_state_dict(torch.load(p_target, map_location=map_loc))

        if os.path.exists(p_opt):
            try:
                opt_state = torch.load(p_opt, map_location=map_loc)
                self.optimizer.load_state_dict(opt_state)
            except Exception:
                pass

        with open(p_meta, "r") as f:
            meta = json.load(f)

        if load_replay_buffer and os.path.exists(p_replay):
            with open(p_replay, "rb") as f:
                data = pickle.load(f)
            if self.use_prioritizedRB:
                self.memory.load(data)
            else:
                self.memory = deque(data, maxlen=self.memory_size)

        print("✅ Modelo cargado correctamente.")
        return meta

    # -------------------------
    # utility
    # -------------------------
    def to(self, device: torch.device):
        self.device = device
        self.policy_net.to(device)
        self.target_net.to(device)
        # optimizer may require re-creating for some devices - keep as is usually OK

    def set_eval(self):
        self.policy_net.eval()
        self.target_net.eval()

    def set_train(self):
        self.policy_net.train()
        self.target_net.train()


# End of file
