import os
import json
import pickle
import torch
import random
import numpy as np
from collections import deque
from NNBase import DQN, errores, optimizadores


class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        hidden_layers=[64, 64],
        lr=1e-3,
        gamma=0.99,

        epsilon=1.0,
        epsilon_decay=0.99995,
        epsilon_min=0.05,

        batch_size=64,
        memory_size=10000,
        update_target_every=1000,
        tau=0.0,

        hidden_activation="relu",
        output_activation="identity",
        loss_function="mse",
        optimizer="adam",

        use_prioritizedRB=False,
        use_noisy=False,
        use_softmax_action=False,
        softmax_tau=1.0
    ):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cuda")

        # Hyperparams
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.tau = tau  # soft update coefficient

        # Exploration parameters (ignored with NoisyNet)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.use_noisy = use_noisy
        self.use_softmax_action = use_softmax_action
        self.softmax_tau = softmax_tau

        # Replay memory
        self.use_prioritizedRB = use_prioritizedRB
        self.memory_size = memory_size
        self.memory = deque(maxlen=memory_size)

        # Networks
        self.policy_net = DQN(
            state_size, hidden_layers, action_size,
            hidden_activation, output_activation,
            loss_function, optimizer, lr,
            use_noisy=use_noisy
        )

        self.target_net = DQN(
            state_size, hidden_layers, action_size,
            hidden_activation, output_activation,
            loss_function, optimizer, lr,
            use_noisy=use_noisy
        )
        
        self.policy_net = self.policy_net.to(self.device)
        self.target_net = self.target_net.to(self.device)


        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optimizadores[optimizer](self.policy_net.parameters(), lr=lr)
        self.criterion = errores[loss_function]()

        self.steps = 0
        
    # ============================================================
    # ---------------------- DISABLE NOISY ------------------------
    # ============================================================

    def disable_noisy(self):
        """Desactiva por completo Noisy Networks y su ruido."""

        self.use_noisy = False
        self.epsilon = 0.0 

        # Eliminar el ruido de cualquier capa noisy
        with torch.no_grad():
            for module in self.policy_net.modules():
                if hasattr(module, "reset_noise"):
                    module.reset_noise = lambda: None  # anula el ruido

            for module in self.target_net.modules():
                if hasattr(module, "reset_noise"):
                    module.reset_noise = lambda: None

        print("🔇 Noisy Networks desactivado correctamente.")


    # ============================================================
    # ------------------------- ACTION ----------------------------
    # ============================================================

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Noisy Net: resetea ruido en cada paso
        if self.use_noisy:
            if hasattr(self.policy_net, "reset_noise"):
                self.policy_net.reset_noise()

        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze()

        # Softmax Exploration
        if self.use_softmax_action:
            probs = torch.softmax(q_values / self.softmax_tau, dim=0).cpu().numpy()
            return np.random.choice(self.action_size, p=probs)

        # Epsilon-greedy (si no hay NoisyNet)
        if not self.use_noisy and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        return q_values.argmax().item()

    # ============================================================
    # -------------------- PURE GREEDY ACTION ---------------------
    # ============================================================

    def select_action_greedy(self, state):
        """
        Selección de acción 100% determinista:
        - Ignora epsilon
        - Ignora NoisyNet
        - Ignora softmax
        - No modifica ningún parámetro
        - No hace reset_noise
        - No hace exploración
        """

        # Convertir estado
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        # Forward determinista
        with torch.no_grad():
            q_values = self.policy_net(state_tensor).squeeze()

        # Acción con mayor Q
        return int(q_values.argmax().item())


    # ============================================================
    # ------------------------- MEMORY ----------------------------
    # ============================================================

    def store_transition(self, state, action, reward, next_state, done):
        if not self.use_prioritizedRB:
            self.memory.append((state, action, reward, next_state, done))
        else:
            self.memory.add((state, action, reward, next_state, done), td_error=1.0)


    # ============================================================
    # -------------------- TARGET NETWORK UPDATE ------------------
    # ============================================================

    def _update_target_network(self):
        if self.tau > 0:
            # Soft update
            for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + policy_param.data * self.tau
                )
        else:
            # Hard update
            self.target_net.load_state_dict(self.policy_net.state_dict())


    # ============================================================
    # ------------------------ TRAINING ----------------------------
    # ============================================================

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        # Sample
        if not self.use_prioritizedRB:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            weights = None
            idxs = None
        else:
            batch, idxs, weights = self.memory.sample(self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            weights = torch.tensor(weights, dtype=torch.float32)

        # Tensors
        states = torch.from_numpy(np.asarray(states, dtype=np.float32))
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.from_numpy(np.asarray(next_states, dtype=np.float32))
        dones = torch.tensor(dones, dtype=torch.float32)
        
        print(" ---- DEBUG BATCH ----")
        print("states:", states)
        print("actions:", actions)
        print("next_states:", next_states)
        print("rewards:", rewards)
        print("dones:", dones)
        print("-----------------------")

        # Current Q
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()

        # ---------- DOUBLE DQN ----------
        next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze().detach()

        # Bellman target
        target_q = rewards + self.gamma * next_q_values * (1 - dones)
        td_errors = target_q - q_values

        # Loss
        if self.use_prioritizedRB:
            loss_per_sample = self.criterion(q_values, target_q.detach())
            loss = (loss_per_sample * weights).mean()
            new_td = td_errors.detach().cpu().numpy()
            self.memory.update_priorities(idxs, new_td)
        else:
            loss = self.criterion(q_values, target_q.detach())

        # Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Reset noisy layers after learning
        if self.use_noisy:
            if hasattr(self.policy_net, "reset_noise"):
                self.policy_net.reset_noise()
            if hasattr(self.target_net, "reset_noise"):
                self.target_net.reset_noise()

        # Update exploration (if not Noisy)
        if not self.use_noisy:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        # Target update schedule
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self._update_target_network()
            
        print("Loss:", loss.item())



    # ============================================================
    # ------------------------- SAVE / LOAD -----------------------
    # ============================================================

    def save(self, folder="model", name="dqn_agent", save_replay_buffer=True):
        os.makedirs(folder, exist_ok=True)

        torch.save(self.policy_net.state_dict(), f"{folder}/{name}_policy.pt")
        torch.save(self.target_net.state_dict(), f"{folder}/{name}_target.pt")

        data = {
            "epsilon": self.epsilon,
            "steps": self.steps,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "memory_size": self.memory_size,
            "use_prioritizedRB": self.use_prioritizedRB,
        }

        with open(f"{folder}/{name}_meta.json", "w") as f:
            json.dump(data, f, indent=4)

        if save_replay_buffer:
            with open(f"{folder}/{name}_replay.pkl", "wb") as f:
                if self.use_prioritizedRB:
                    pickle.dump(self.memory, f)
                else:
                    pickle.dump(list(self.memory), f)

        return data


    def load(self, folder="model", name="dqn_agent", load_replay_buffer=True):
        policy_path = f"{folder}/{name}_policy.pt"
        target_path = f"{folder}/{name}_target.pt"
        meta_path = f"{folder}/{name}_meta.json"

        if not (os.path.exists(policy_path) and os.path.exists(target_path)):
            print("⚠️ No se encontró modelo, entrenando desde cero...")
            return

        self.policy_net.load_state_dict(torch.load(policy_path))
        self.target_net.load_state_dict(torch.load(target_path))

        with open(meta_path, "r") as f:
            data = json.load(f)

        if load_replay_buffer:
            replay_path = f"{folder}/{name}_replay.pkl"
            if os.path.exists(replay_path):
                with open(replay_path, "rb") as f:
                    replay_data = pickle.load(f)
                if self.use_prioritizedRB:
                    self.memory = replay_data
                else:
                    self.memory = deque(replay_data, maxlen=self.memory_size)

        print("✅ Modelo cargado correctamente.")
        return data
