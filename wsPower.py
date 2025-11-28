# app.py
import json
import asyncio
import time
from typing import Dict, List

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

# Importa tu agente DQN aquí
from DQNAgentGPU import DQNAgent  # debe tener: select_action(state), store_transition(s,a,r,ns,d), train_step(), save(), load()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================
GLOBAL_EPISODES_TO_WAIT = 100        # cuando TODOS los agentes alcanzan N episodios -> evaluación global
GLOBAL_EVAL_MIN_EPISODES_FOR_AVG10 = 10

TRAIN_INTERVAL_STEPS = 200           # cada N transiciones recibidas -> lanzar entrenamiento en batch
TRAIN_STEPS_PER_INTERVAL = 50        # cuántos pasos de entrenamiento ejecutar en cada intervalo (llamadas a train_step)
MAX_BATCH_SELECT_ACTION = 2048       # límite para número de acciones a computar por batch (por seguridad)

SAVE_MODEL_EVERY = 500               # Cada cuanto episodios guardara el modelo
global_episode_counter = 0

# ============================================================
# ESTADO GLOBAL / ESTRUCTURAS
# ============================================================
next_agent_id = 0                     # asigna ids a agentes que pidan -1
agents_ws: Dict[int, WebSocket] = {}  # agent_id -> websocket (solo si quieres trackear)
connected_agents: Dict[int, float] = {}  # agent_id -> last_seen timestamp

# tracking por agente (dinámico)
episode_counts: Dict[int, int] = {}      # agent_id -> episodios completados en ciclo actual
final_distances: Dict[int, List[float]] = {}  # agent_id -> lista de distancias finales
episode_rewards: Dict[int, float] = {}   # reward acumulado del episodio actual (opcional)

# Contadores para entrenamiento
received_transitions = 0
train_lock = asyncio.Lock()
global_lock = asyncio.Lock()   # para guardados/clonados/evaluaciones

# ============================================================
# CREAR AGENTE GLOBAL (UN SOLO MODELO PARA TODOS)
# ============================================================
def create_agent() -> DQNAgent:
    agent = DQNAgent(
        state_size=7,
        action_size=4,
        hidden_layers=[128,128],
        use_noisy=True,              # <--- activar
        epsilon=0.0,                 # <--- con NoisyNet NO se usa epsilon
        epsilon_decay=1.0,           # <--- ignóralo
        epsilon_min=0.0,             # <--- ignóralo
    )
    # intenta cargar checkpoint base si existe (opcional)
    #try:
    #    
    #    agent.load(folder="models", name="a", load_replay_buffer=False)
    #    # agent.disable_noisy()
    #except Exception:
    #    pass
    return agent

global_agent = create_agent()

# histórico global
best_global_avg10 = float("inf")
best_global_agent = None  # agent id that produced best_global_avg10 historically

# ============================================================
# UTILIDADES
# ============================================================
def safe_as_float_array(x) -> np.ndarray:
    # convierte listas / nested lists en np.float32 arrays
    return np.asarray(x, dtype=np.float32)

async def try_train_steps(n_steps: int):
    """
    Ejecutar train_step n_steps veces de forma no bloqueante.
    El DQNAgent.train_step() debería tomar el batch desde su replay interno.
    """
    async with train_lock:
        for _ in range(n_steps):
            try:
                # global_agent.store_transition()
                global_agent.train_step()
            except Exception as e:
                # captura errores no fatales
                print("Error during train_step:", e)
                return

async def evaluate_and_save_global_if_ready():
    """
    Si todos los agentes actualmente conectados alcanzaron GLOBAL_EPISODES_TO_WAIT,
    evaluar avg10 por agente (si tienen suficientes episodios), escoger mejor candidato,
    y si mejora el histórico, guardar best_global_model y clonar.
    Luego reiniciar episode_counts y final_distances para el siguiente ciclo.
    """
    global best_global_avg10, best_global_agent

    async with global_lock:
        # no hay agentes -> nada que hacer
        if len(connected_agents) == 0:
            return

        # Revisar si todos alcanzaron el umbral
        for ag in list(connected_agents.keys()):
            if episode_counts.get(ag, 0) < GLOBAL_EPISODES_TO_WAIT:
                return  # al menos uno no llegó

        print("🔎 Todos los agentes alcanzaron el umbral. Evaluando candidatos...")

        best_candidate = None
        best_candidate_avg = float("inf")

        # buscar mejor avg10 entre agentes presentes
        for ag, fdist in final_distances.items():
            if ag not in connected_agents:
                continue
            if len(fdist) >= GLOBAL_EVAL_MIN_EPISODES_FOR_AVG10:
                last_n = fdist[-10:] if len(fdist) >= 10 else fdist
                avg10 = float(sum(last_n) / len(last_n))
                print(f"  - Agente {ag} AVG10 = {avg10:.4f} (n={len(last_n)})")
                if avg10 < best_candidate_avg:
                    best_candidate_avg = avg10
                    best_candidate = ag
            else:
                print(f"  - Agente {ag} no califica para AVG10 (tiene {len(fdist)})")

        if best_candidate is None:
            print("⚠️ Ningun candidato con suficientes episodios. Reiniciando conteos para evitar bloqueo.")
            # Reiniciar y continuar
            for ag in list(connected_agents.keys()):
                episode_counts[ag] = 0
                final_distances[ag] = []
            return

        # Si mejora al histórico, guardar y clonar a todos
        if best_candidate_avg < best_global_avg10:
            best_global_avg10 = best_candidate_avg
            best_global_agent = best_candidate
            print(f"🏆 Nuevo mejor modelo GLOBAL: agente {best_candidate} AVG10={best_candidate_avg:.4f}")

            # Guardar modelo global (usa DQNAgent.save)
            try:
                global_agent.save(folder="models", name="best_global_model")
                print("💾 best_global_model guardado.")
            except Exception as e:
                print("❌ Error guardando best_global_model:", e)

            # Clonar: en este diseño todos usan el mismo global_agent,
            # así que no es necesario 'cargar' en otros agentes a nivel servidor.
            # Si necesitas notificar a Unity que debe recargar algo, podrías notificarlo.
        else:
            print(f"ℹ️ Ningún candidato superó histórico ({best_global_avg10:.4f}). Mejor candidato: {best_candidate_avg:.4f}")

        # Reiniciar para siguiente ciclo
        for ag in list(connected_agents.keys()):
            episode_counts[ag] = 0
            final_distances[ag] = []

episode_rewards_history = {}

# ============================================================
# WEBSOCKET BATCH ENDPOINT: recibe lista de agentes y devuelve acciones
# ============================================================
@app.websocket("/ws_batch")
async def websocket_batch(ws: WebSocket):
    global next_agent_id, received_transitions, global_episode_counter

    await ws.accept()
    client_id = f"client_{id(ws)}"
    print(f"🟢 WebSocket batch conectado: {client_id}")
    
    # global_agent.policy_net.train()
    # global_agent.enable_noisy()
    # global_agent.set_train()

    try:
        while True:
            # Espera texto (JSON) del cliente (Unity)
            msg = await ws.receive_text()
            data = json.loads(msg)

            # Mensaje esperado:
            # { "agents": [ { "id": -1 | int, "state": [...], "reward": float, "next_state": [...], "done": bool, "distance": float(optional) }, ... ] }
            agents_list = data.get("agents", [])

            # RESPUESTAS que enviaremos de vuelta
            actions_resp = []

            # Pre-collect states to maybe batch-predict later
            states_for_batch = []
            ids_for_batch = []
            raw_entries = []  # keep raw entries to store transitions after actions

            for entry in agents_list:
                aid = int(entry.get("id", -1))

                # si cliente pide id -1 -> asignar uno nuevo automáticamente
                if aid == -1:
                    aid = next_agent_id
                    next_agent_id += 1

                # registrar
                connected_agents[aid] = time.time()
                agents_ws[aid] = ws
                episode_counts.setdefault(aid, 0)
                final_distances.setdefault(aid, [])
                episode_rewards.setdefault(aid, 0.0)

                # ====== estado actual → es el next_state real ======
                state = safe_as_float_array(entry.get("state", []))
                states_for_batch.append(state)
                ids_for_batch.append(aid)

                # ====== TRANSICIÓN COMPLETA (Unity ya la arma) ======
                prev_state = safe_as_float_array(entry.get("prev_state", None))
                prev_action = entry.get("prev_action", None)
                prev_reward = entry.get("prev_reward", None)
                prev_done = entry.get("prev_done", None)

                # guardar crudo para debug
                raw_entries.append({
                    "id": aid,
                    "state": state,          # next_state
                    "prev_state": prev_state,
                    "prev_action": prev_action,
                    "prev_reward": prev_reward,
                    "prev_done": prev_done,
                    "done": bool(entry.get("done", False)),
                    "reward": float(entry.get("reward", 0.0)),
                    "distance": float(entry.get("distance", 0.0)) if entry.get("distance", None) is not None else None
                })


            # ---------------------------
            # Selección de acciones (batch-friendly)
            # ---------------------------
            # Intentamos usar selección por lotes si el agente tiene un método; en caso contrario,
            # iteramos (seguimos siendo razonables para cientos/agentes).
            actions_list = []

            # Si tu DQNAgent tiene método 'select_actions_batch(states_np)' lo usarías aquí.
            # Para compatibilidad general usamos iteración (puedes modificar si tienes batch predict).
            for s in states_for_batch:
                try:
                    a = global_agent.select_action(s)
                except Exception:
                    # fallback seguro
                    a = int(np.random.randint(0, global_agent.action_size))
                actions_list.append(int(a))

            # Construir respuesta
            for aid, act in zip(ids_for_batch, actions_list):
                actions_resp.append({"id": aid, "action": int(act)})

            # ---------------------------
            # Enviar acciones de vuelta al client (Unity)
            # ---------------------------
            await ws.send_text(json.dumps({"actions": actions_resp}))

            # ---------------------------
            # Registrar transiciones en replay y contadores:
            # para cada raw_entry, almacenar (s,a,r,ns,done) en replay buffer del global_agent
            # ---------------------------
            for entry, act in zip(raw_entries, actions_list):
                aid = entry["id"]

                # estado actual (next_state real)
                state = entry["state"]

                # Datos de la transición previa (si existen)
                prev_state = entry.get("prev_state", None)
                prev_action = entry.get("prev_action", None)
                prev_reward = entry.get("prev_reward", None)
                prev_done = entry.get("prev_done", None)

                # La acción que enviamos ahora al agente (para este state)
                # la usamos solo para enviar al cliente, no para la transición previa
                action_to_send = int(act)

                # Enviar acción (ya lo haces más arriba con actions_resp)
                # ---------------------------
                # Guardar la transición previa SOLO si está completa
                if prev_state is not None and prev_action is not None:
                    # Convierte tipos y guarda en replay
                    try:
                        global_agent.store_transition(
                            prev_state,
                            int(prev_action),
                            float(prev_reward) if prev_reward is not None else 0.0,
                            state,                  # next_state real
                            bool(prev_done) if prev_done is not None else False
                        )
                    except Exception as e:
                        try:
                            global_agent.store_transition(
                                np.asarray(prev_state, dtype=np.float32),
                                int(prev_action),
                                float(prev_reward) if prev_reward is not None else 0.0,
                                np.asarray(state, dtype=np.float32),
                                bool(prev_done) if prev_done is not None else False
                            )
                        except Exception as e2:
                            print("Error store_transition:", e, e2)

                # Tracking de reward/episodios usando reward/done actuales (si quieres)
                # Nota: entry['reward'] corresponde al reward acumulado desde el último tick (current)
                reward_now = float(entry.get("reward", 0.0))
                done_now = bool(entry.get("done", False))
                episode_rewards[aid] = episode_rewards.get(aid, 0.0) + reward_now

                # Si el episodio terminó en este tick (done_now), registra y resetea contadores
                if done_now:
                    # Guarda reward final del episodio (acumulado)
                    final_reward = episode_rewards.get(aid, 0.0)

                    # === PROMEDIO 100 EPISODIOS (RewardAvg100) ===
                    rewards_list = episode_rewards_history.setdefault(aid, [])
                    rewards_list.append(final_reward)

                    avg100 = np.mean(rewards_list[-100:])
                    print(f"[RewardAvg100] Agent {aid} → {avg100:.2f}")

                    # === distancias finales ===
                    distance = entry.get("distance", None)
                    if distance is None and len(state) > 2:
                        dist_val = float(state[2])
                    else:
                        dist_val = float(distance) if distance is not None else 0.0

                    final_distances.setdefault(aid, []).append(dist_val)
                    episode_counts[aid] = episode_counts.get(aid, 0) + 1
                    episode_rewards[aid] = 0.0

                    global_episode_counter += 1
                    if global_episode_counter % SAVE_MODEL_EVERY == 0:
                        async with global_lock:
                            try:
                                fname = f"global_every_{SAVE_MODEL_EVERY}_global_ep"
                                global_agent.save(folder="models", name=fname)
                                print(f"💾 Guardado periódico cada {SAVE_MODEL_EVERY} episodios.")
                            except Exception as e:
                                print("❌ Error al guardar modelo periódicamente:", e)


            # ---------------------------
            # Gestión de entrenamiento por intervalos
            # ---------------------------
            received_transitions += len(raw_entries)
            # Lanzar entrenamiento asíncrono si pasamos el umbral
            if received_transitions >= TRAIN_INTERVAL_STEPS:
                # reseteamos el contador localmente y lanzamos entrenamiento
                received_transitions = 0
                # lanzamos en background (no bloqueamos el socket)
                asyncio.create_task(try_train_steps(TRAIN_STEPS_PER_INTERVAL))

            # ---------------------------
            # Intentar evaluación global si corresponde
            # ---------------------------
            await evaluate_and_save_global_if_ready()

    except WebSocketDisconnect:
        print(f"🟠 Cliente batch {client_id} desconectado.")
        # No sabemos qué agentes estaban en este cliente; mantenemos registros por last_seen
        return
    except Exception as e:
        print("❌ Error en websocket_batch loop:", e)
        return

# ============================================================
# ENDPOINTS ADICIONALES (monitoring)
# ============================================================
@app.get("/status")
async def status():
    return {
        "connected_agents": list(connected_agents.keys()),
        "n_connected": len(connected_agents),
        "best_global_avg10": best_global_avg10,
        "best_global_agent": best_global_agent
    }

# ============================================================
# INICIALIZADOR: training loop periódico (opcional)
# ============================================================
async def periodic_train_loop():
    """
    Un loop opcional que garantiza entrenamiento periódico aun cuando
    no se alcance TRAIN_INTERVAL_STEPS (por si tu flujo es bajo).
    """
    while True:
        await asyncio.sleep(5.0)
        # Ejecuta unos pasos mínimos si hay datos en replay
        
        # 🔍 Debug rápido del ruido NoisyNet
        # if global_agent.use_noisy:
        #     global_agent.debug_noisy()
            
            
        try:
            asyncio.create_task(try_train_steps(5))
        except Exception:
            pass

# Lanzar el loop en background cuando arranque FastAPI
@app.on_event("startup")
async def startup_event():
    print("🚀 Servidor arrancando. Inicializando periodic_train_loop...")
    asyncio.create_task(periodic_train_loop())

# ============================================================
# WEBSOCKET DE PRUEBA INDIVIDUAL: SOLO INFERENCIA (SIN ENTRENAR)
# ============================================================
@app.websocket("/ws_test")
async def websocket_test(ws: WebSocket):
    await ws.accept()
    print("🧪 WebSocket de prueba conectado")

    # 🔒 Modo inferencia determinista
    global_agent.disable_noisy()
    global_agent.policy_net.eval()

    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)

            agents_list = data.get("agents", [])
            if len(agents_list) == 0:
                await ws.send_text(json.dumps({"error": "No se envió ningún agente"}))
                continue

            # Solo tomamos EL PRIMER agente
            agent = agents_list[0]

            aid = int(agent.get("id", -1))
            if aid == -1:
                aid = 999999  # id fijo para test solamente

            # Convertimos el state correctamente
            state = safe_as_float_array(agent.get("state", []))

            # Acción 100% determinista
            try:
                action = int(global_agent.select_action_greedy(state))
            except Exception:
                action = int(np.random.randint(0, global_agent.action_size))

            # Respuesta usando el MISMO formato que /ws_batch
            await ws.send_text(json.dumps({
                "actions": [
                    {
                        "id": aid,
                        "action": action
                    }
                ]
            }))

    except WebSocketDisconnect:
        print("🧪 Cliente de prueba desconectado")

    except Exception as e:
        print("❌ Error en ws_test:", e)

