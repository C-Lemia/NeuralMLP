import modelo_logico  # importa o script com o treinamento
import tensorflow as tf
import numpy as np
import random

modelo_logico.treinar_modelo() #------------- Treina e salva o modelo 


model = tf.keras.models.load_model("modelo_logico.keras")#---------------- Carrega o modelo salvo

#---------- Função de decisão da IA
def logica_jogo(energia, inimigo_perto): #---------------- energia (0 ou 1) inimigo próximo (0 ou 1)
    entrada = np.array([[energia, inimigo_perto]], dtype=np.float32) #-------------- se energia=1 e inimigo_perto=0, então entrada = [[1.0, 0.0]]
    saida = model.predict(entrada, verbose=0)[0] #-------------- será uma lista com 3 valores entre 0 e 1 (AND, OR, XOR)
    print(f"DEBUG: entrada={entrada[0]} → saida={saida}")
    saida_bin = (saida > 0.5).astype(int) #---------------- se valor > 0.5 → vira 1 e se valor ≤ 0.5 → vira 0

    acao = [] #----------- cada saída ativa uma ação no jogo
    if saida_bin[0]: acao.append("⚔️ Atacar")
    if saida_bin[1]: acao.append("🛡️ Defender")
    if saida_bin[2]: acao.append("🏃 Correr")

    return acao if acao else ["😴 Esperar"]

#-------------- Simula 8 turnos
print("=== SIMULAÇÃO DO JOGO ===")
for _ in range(8):#--------- simulando 8 rodadas de decisões no jogo, loop
    energia = random.randint(0, 1) #------ se o jogador está com energia (1) ou não (0)
    inimigo = random.randint(0, 1) #------------ se há um inimigo próximo (1) ou não (0)
    acao = logica_jogo(energia, inimigo)
    print(f"[energia={energia} | inimigo_perto={inimigo}] → Ação: {', '.join(acao)}")
