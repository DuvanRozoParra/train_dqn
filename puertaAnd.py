import random
import numpy as np
from DQNAgent import DQNAgent

class AndEnv:
    def __init__(self):
        self.state = None
        
    def reset(self):
        self.x1 = random.randint(0,1)
        self.x2 = random.randint(0,1)
        self.state = np.array([self.x1,self.x2], dtype= np.float32)
        return self.state
    
    def flujo(self, action):
        action = int(action)
        
        respuesta_correcta = self.x1 & self.x2
        
        recompensa = 1 if action == respuesta_correcta else -1
        
        done = True
        
        next_state = self.reset()
        
        return next_state, recompensa, done, {}
    
ia = DQNAgent(
    state_size=2,
    action_size=2,
    hidden_layers=[168,168,168,168,168],
    gamma=0.99,
    epsilon_decay=0.995,
    epsilon_min=0.01,
    lr=1e-3,
    use_noisy=False,
    use_softmax_action=False
)

env = AndEnv()
episodios = 1000

for ep in range(episodios):
    state = env.reset()
    
    action = ia.select_action(state)
    next_state, reward, done, _ = env.flujo(action)

    ia.store_transition(state, action, reward, next_state, done)
    ia.train_step()
    
    if ep % 200 == 0:
        print(f"Episodio {ep} — recompensa={reward}, epsilon={ia.epsilon:.3f}")
        
print("entrenamiento terminado.")

test1 = np.array([1,1], dtype= np.float32)
test2 = np.array([1,0], dtype= np.float32)
test3 = np.array([0,1], dtype= np.float32)
test4 = np.array([0,0], dtype= np.float32)

iaPreview1 = ia.select_action(test1)
iaPreview2 = ia.select_action(test2) 
iaPreview3 = ia.select_action(test3) 
iaPreview4 = ia.select_action(test4) 

print(f"Entra: {test1} respuesta ia: {iaPreview1}")
print(f"Entra: {test2} respuesta ia: {iaPreview2}")
print(f"Entra: {test3} respuesta ia: {iaPreview3}")
print(f"Entra: {test4} respuesta ia: {iaPreview4}")

