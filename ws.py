import json
import os
import asyncio
import numpy as np
from fastapi import WebSocket, FastAPI
from fastapi.websockets import WebSocketDisconnect
from DQNAgent import DQNAgent

app = FastAPI()

# ============================================
# AGENTES DINÁMICOS POR CONEXIÓN
# ============================================
next_agent_id = 0               # contador global para asignar IDs
agents = {}                     # agent_id -> DQNAgent
episode_rewards = {}            # agent_id -> recompensa acumulada del episodio actual
episode_counts = {}             # agent_id -> episodios terminados (para el ciclo actual)
final_distances = {}            # agent_id -> lista de distancias finales por episodio

# ============================================
# PARÁMETROS GLOBALES DE SINCRONIZACIÓN
# ============================================
GLOBAL_EPISODES_TO_WAIT = 100   # N episodios por agente antes de evaluar globalmente
GLOBAL_EVAL_MIN_EPISODES_FOR_AVG10 = 10  # mínimo episodios para calcular AVG10

# Lock para sincronizar evaluación / guardado / clonación global
global_lock = asyncio.Lock()

# ============================================
# MEJOR MODELO GLOBAL
# ============================================
best_global_avg10 = float("inf")   # mejor avg10 histórico global
best_global_agent = None           # id del agente que produjo ese mejor avg10


# ============================================
# FUNCIONES AUXILIARES
# ============================================

def create_agent():
    """
    Crea e inicializa un agente DQN con la configuración base.
    Ajusta esta función según tu DQNAgent.
    """
    agent = DQNAgent(
        state_size=7,
        action_size=4,
        hidden_layers=[64, 128, 128, 64],
        gamma=0.99,
        epsilon=0.0,
        epsilon_decay=1.0,
        epsilon_min=0.0,
        use_noisy=True,
        use_softmax_action=False,
        memory_size=1_000_000,
        optimizer="adam",
        loss_function="huber",
        lr=1e-3,
        update_target_every=500,
        tau=0.0,
        use_prioritizedRB=False
    )

    # Cargar un checkpoint base si lo deseas (opcional)
    try:
        agent.load(folder="best", name="a", load_replay_buffer=False)
        # agent.disable_noisy()
    
    except Exception as e:
        print("ℹ️ No se pudo cargar agente base (ignore si no existe):", e)

    return agent


async def evaluate_and_save_global_if_ready():
    """
    Revisar si TODOS los agentes actualmente conectados alcanzaron
    GLOBAL_EPISODES_TO_WAIT. Si es así, evaluar avg10 de cada agente,
    elegir el mejor, guardar como best_global_model y clonar a todos.
    Luego reiniciar episode_counts y final_distances para el siguiente ciclo.
    """
    global best_global_avg10, best_global_agent

    async with global_lock:
        # Si no hay agentes conectados, no hacer nada
        if len(agents) == 0:
            return

        # Revisar si TODOS los agentes alcanzaron el umbral
        for ag_id in list(agents.keys()):
            if episode_counts.get(ag_id, 0) < GLOBAL_EPISODES_TO_WAIT:
                # Al menos uno no ha llegado, salir
                return

        # Todos alcanzaron el umbral => Evaluar candidatos
        best_candidate = None
        best_candidate_avg = float("inf")

        print("🔎 Todos los agentes alcanzaron el umbral. Evaluando AVG10 por agente...")

        for ag_id, fdist in final_distances.items():
            # Solo evaluar agentes que aún existen y tienen suficientes episodios
            if ag_id not in agents:
                continue
            if len(fdist) >= GLOBAL_EVAL_MIN_EPISODES_FOR_AVG10:
                avg10 = sum(fdist[-10:]) / 10 if len(fdist) >= 10 else sum(fdist) / len(fdist)
                print(f"  - Agente {ag_id}: AVG10 (últimas 10) = {avg10:.4f}")
                if avg10 < best_candidate_avg:
                    best_candidate_avg = avg10
                    best_candidate = ag_id
            else:
                # Si no tiene suficientes episodios para AVG10, lo consideramos con avg muy malo (skip)
                print(f"  - Agente {ag_id}: NO tiene suficientes episodios para AVG10 (tiene {len(fdist)})")

        if best_candidate is None:
            print("⚠️ Ningún candidato con suficientes episodios para evaluar AVG10. Reiniciando conteos y continuando.")
            # Reiniciamos para evitar bloqueo perpetuo (opcional; puedes preferir esperar más)
            for ag_id in list(agents.keys()):
                episode_counts[ag_id] = 0
                final_distances[ag_id] = []
            return

        # Si el candidato es mejor que el global histórico, guardar y clonar
        if best_candidate_avg < best_global_avg10:
            print(f"\n🏆 Nuevo MEJOR MODELO GLOBAL: agente {best_candidate} con AVG10 = {best_candidate_avg:.4f}")
            best_global_avg10 = best_candidate_avg
            best_global_agent = best_candidate

            # Guardar modelo (usa tu método save)
            try:
                agents[best_candidate].save(folder="models", name="best_global_model")
                print("💾 best_global_model guardado.")
            except Exception as e:
                print("❌ Error guardando el best_global_model:", e)

            # Clonar el modelo guardado a todos los agentes
            for other_id, other_agent in agents.items():
                if other_id == best_candidate:
                    continue
                try:
                    other_agent.load(folder="models", name="best_global_model", load_replay_buffer=False)
                except Exception as e:
                    print(f"❌ Error clonando modelo a agente {other_id}:", e)

        else:
            print(f"ℹ️ Ningún modelo actual superó el histórico global ({best_global_avg10:.4f}). Mejor candidato tuvo {best_candidate_avg:.4f}")

        # Reiniciar conteos y listas para el siguiente ciclo
        print("🔁 Reiniciando episode_counts y final_distances para el siguiente ciclo global.")
        for ag_id in list(agents.keys()):
            episode_counts[ag_id] = 0
            final_distances[ag_id] = []


# ============================================
# WEBSOCKET: CADA CONEXIÓN CREA SU AGENTE
# ============================================
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global next_agent_id

    await ws.accept()

    # Asignar ID único
    agent_id = next_agent_id
    next_agent_id += 1

    # Crear agente y registros
    agent = create_agent()
    agents[agent_id] = agent
    episode_rewards[agent_id] = 0.0
    episode_counts[agent_id] = 0
    final_distances[agent_id] = []

    print(f"🟢 Nuevo agente conectado → ID {agent_id}  (total agentes: {len(agents)})")

    # Enviar el ID al cliente Unity
    try:
        await ws.send_text(json.dumps({"agent_id_assigned": agent_id}))
    except Exception as e:
        print("❌ Error enviando agent_id al cliente:", e)

    # Opcional: intentar cargar un checkpoint inicial si lo deseas
    try:
        agent.load(folder="models", name="agent_0_epi_380", load_replay_buffer=False)
    except Exception:
        pass

    try:
        while True:
            msg = await ws.receive()

            raw_bytes = msg.get("bytes")
            raw_text = msg.get("text")

            try:
                data = json.loads(raw_bytes.decode("utf-8") if raw_bytes else raw_text)
            except Exception:
                await ws.send_text("Error: JSON inválido")
                continue

            # Lectura de datos enviados por Unity
            aid = int(data["agent_id"])
            # seguridad: si aid no coincide con agent_id, usar el agent_id asignado
            if aid != agent_id:
                print(f"⚠️ agent_id recibido ({aid}) no coincide con asignado ({agent_id}), usando asignado.")
                aid = agent_id

            # Recuperar el agente actual (por si hubo recarga)
            agent = agents.get(aid)
            if agent is None:
                await ws.send_text(json.dumps({"error": "agent_not_found"}))
                continue

            state = np.array(data["state"], dtype=np.float32)
            reward = float(data["reward"])
            next_state = np.array(data["next_state"], dtype=np.float32)
            done = bool(data["done"])

            # 1) Seleccionar acción
            action = agent.select_action(state)

            await ws.send_text(json.dumps({
                "agent_id": aid,
                "action": int(action)
            }))

            # 2) Tracking de reward del episodio
            episode_rewards[aid] = episode_rewards.get(aid, 0.0) + reward

            # 3) Si finaliza episodio -> actualizar contadores y distancias
            if done:
                # Asumo que la distancia final está en state[2] (igual que tu código)
                distance = float(state[2])
                final_distances.setdefault(aid, []).append(distance)
                episode_counts[aid] = episode_counts.get(aid, 0) + 1

                epi = episode_counts[aid]
                print(f"🏁 Fin episodio {epi} del Agente {aid}  (distancia final: {distance:.4f})")

                # Guardar mejor por episodio local (opcional)
                # -- si quieres guardar el mejor reward por agente local, puedes hacerlo aquí

                # Reiniciar reward acumulado del episodio
                episode_rewards[aid] = 0.0

                # Tras cada episodio, intentar evaluar/clonar global si TODOS los agentes alcanzaron el umbral
                await evaluate_and_save_global_if_ready()

            # 4) Almacenar transición y entrenar
            try:
                agent.store_transition(state, action, reward, next_state, done)
                agent.train_step()
            except Exception as e:
                print("❌ Error en store/train_step:", e)

    except WebSocketDisconnect:
        # Cliente se desconectó: limpiar registros de este agente
        print(f"🔴 Agente {agent_id} se desconectó.")
        # Asegurarse de eliminar claves con protección
        agents.pop(agent_id, None)
        episode_rewards.pop(agent_id, None)
        episode_counts.pop(agent_id, None)
        final_distances.pop(agent_id, None)

        # Nota: no reiniciamos best_global_* (histórico)
        # Si quieres, podríamos re-evaluar el estado global tras la desconexión:
        # await evaluate_and_save_global_if_ready()
        return

    except Exception as ex:
        print("❌ Error inesperado en websocket_endpoint:", ex)
        # Limpiar (intentar)
        agents.pop(agent_id, None)
        episode_rewards.pop(agent_id, None)
        episode_counts.pop(agent_id, None)
        final_distances.pop(agent_id, None)
        try:
            await ws.close()
        except Exception:
            pass
        return
