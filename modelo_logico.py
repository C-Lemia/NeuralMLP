import tensorflow as tf
import numpy as np

def treinar_modelo():
    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32) # entradas possíveis de dois bits
    y = np.array([
        [0, 0, 0],  #combinação de entrada lógica
        [0, 1, 1],
        [0, 1, 1],
        [1, 1, 0]
    ], dtype=np.float32)

    model = tf.keras.Sequential([ # relu > se x for negativo, o resultado é 0 , se x for positivo, o resultado é ele mesmo,só ativa quando a entrada é maior que 0
        tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)), # camada 1: 8 neurônios, ativação relu, recebe 2 valores (entrada)
                                                                       # shape 2 : dois valores de entrada, por exemplo, [0, 1]
                                                                       # 8 : entre 2x e 4x o número de entradas na primeira camada oculta para permitir que a rede aprenda representações mais complexas
        tf.keras.layers.Dense(6, activation='relu'),# camada 2: 6 neurônios, também com relu, omo se a rede estivesse “resumindo” o que aprendeu na primeira camada para decidir o que ativar ou não na próxima
        tf.keras.layers.Dense(3, activation='sigmoid') # camada final: 3 neurônios com sigmoid, porque queremos 3 saídas entre 0 e 1, uma para cada função lógica: AND, OR, XOR
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])# adam otimizador ajusta os pesos da rede durante o treinamento para minimizar o erro
                                                                                     # binary_crossentropy, calcular o quão errado o modelo está
                                                                                     # acurácia, ão interfere no aprendizado, mas aparece no .fit() e nos logs, quantas vezes o modelo acerta a saída esperada
    model.fit(X, y, epochs=5000, verbose=0) # entrada e resultado esperado (x,y), a rede vai passar 5000 vezes por todos os dados de entrada( testei com menos, mas o aprendizado não estava bom)
    model.save("modelo_logico.keras") # salvar
    print("✅ Modelo treinado e salvo.\n") # mostrar saída

    # Impressão dos resultados aprendidos
    print("=== RESULTADOS APRENDIDOS ===")
    funcoes = ["AND", "OR", "XOR"]
    predicoes = model.predict(X, verbose=0) # são os valores que o modelo calcula para cada entrada
    arredondado = (predicoes > 0.5).astype(int) # transforma em 0 ou 1, com base em 0.5 (limiar)

    for i, entrada in enumerate(X):
        print(f"Entrada: {entrada}") # imprime a entrada atual da rede (por exemplo, [1. 0.])
        for j in range(3): # retorna 3 valores (AND, OR, XOR), então j vai de 0 a 2
            print(f"  {funcoes[j]}: {predicoes[i][j]:.2f} → classificado como {arredondado[i][j]}") # nome da função lógica,predicoes[i][j]:.2f valor previsto pela rede (entre 0 e 1) elimita a saída para 2 casas decimais (por exemplo, 0.97)
        print()

# Só executa se rodar diretamente o script
if __name__ == "__main__":
    treinar_modelo()
