import modelo_logico               #------------ Importa e treina o modelo lógico
import tensorflow as tf
import numpy as np
import random

modelo_logico.treinar_modelo()    #------------ Treina e salva o modelo

#------------ Carrega o modelo treinado
modelo = tf.keras.models.load_model("modelo_logico.keras")

#------------ Função de decisão da IA com base na entrada do jogo
def decidir_acao_jogo(energia, inimigo_proximo):
    entrada_rede = np.array([[energia, inimigo_proximo]], dtype=np.float32)
    saida_rede = modelo.predict(entrada_rede, verbose=0)[0]

    print(f"DEBUG: entrada={entrada_rede[0]} → saída={saida_rede}")

    saida_binaria = (saida_rede > 0.5).astype(int)

    acoes = []
    if saida_binaria[0]:
        acoes.append("⚔️ Atacar")
    if saida_binaria[1]:
        acoes.append("🛡️ Defender")
    if saida_binaria[2]:
        acoes.append("🏃 Correr")

    return acoes if acoes else ["😴 Esperar"]

#------------ Simula 8 rodadas do jogo
print("=== SIMULAÇÃO DO JOGO ===")
for rodada in range(8):
    energia = random.randint(0, 1)         # 1 = com energia, 0 = sem energia
    inimigo_proximo = random.randint(0, 1) # 1 = há inimigo, 0 = sem inimigo

    acoes_tomadas = decidir_acao_jogo(energia, inimigo_proximo)
    print(f"[energia={energia} | inimigo_perto={inimigo_proximo}] → Ação: {', '.join(acoes_tomadas)}")
