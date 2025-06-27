import tensorflow as tf
import numpy as np

def treinar_modelo():
    #------------ Entradas possíveis de dois bits (com 0 e 1)
    entradas = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)

    #------------ Saídas esperadas: [AND, OR, XOR] para cada entrada
    saidas_esperadas = np.array([
        [0, 0, 0],
        [0, 1, 1],
        [0, 1, 1],
        [1, 1, 0]
    ], dtype=np.float32)

    #------------ Definindo o modelo sequencial
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu', input_shape=(2,)),  # camada oculta 1
        tf.keras.layers.Dense(6, activation='relu'),                    # camada oculta 2
        tf.keras.layers.Dense(3, activation='sigmoid')                 # camada de saída (AND, OR, XOR)
    ])

    #------------ Compilando o modelo com otimizador e função de perda
    modelo.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    #------------ Treina o modelo com 5000 épocas
    modelo.fit(entradas, saidas_esperadas, epochs=5000, verbose=0)

    #------------ Salva o modelo treinado
    modelo.save("modelo_logico.keras")
    print("✅ Modelo treinado e salvo.\n")

    #------------ Exibe os resultados aprendidos
    print("=== RESULTADOS APRENDIDOS ===")
    nomes_funcoes = ["AND", "OR", "XOR"]
    predicoes = modelo.predict(entradas, verbose=0)
    predicoes_binarias = (predicoes > 0.5).astype(int)

    for i, entrada in enumerate(entradas):
        print(f"Entrada: {entrada}")
        for j in range(3):
            print(f"  {nomes_funcoes[j]}: {predicoes[i][j]:.2f} → classificado como {predicoes_binarias[i][j]}")
        print()

#------------ Executa o treinamento apenas se rodar diretamente o script
if __name__ == "__main__":
    treinar_modelo()
