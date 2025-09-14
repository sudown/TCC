import joblib
import pandas as pd

print("--- VERIFICANDO O MODELO SALVO COM DADOS REAIS ---")

try:
    # 1. Carregar o modelo treinado
    modelo = joblib.load('./models/modelo_phishing.joblib')
    print("Modelo carregado com sucesso!")

    # 2. Carregar o dataset original para pegar um exemplo
    df = pd.read_csv('./data/Phishing_Legitimate_full.csv')
    print("Dataset original carregado para teste.")

    # 3. Pegar um exemplo que sabemos ser LEGÍTIMO (CLASS_LABEL == 1)
    # Vamos pegar o 10º site legítimo do dataset
    exemplo_legitimo = df[df['CLASS_LABEL'] == 1].iloc[10]
    
    # Separar as features do rótulo verdadeiro
    features_reais = exemplo_legitimo.drop(['CLASS_LABEL', 'id'])
    rotulo_verdadeiro = exemplo_legitimo['CLASS_LABEL']

    # Ajustar o formato para a previsão (precisa ser um array 2D)
    features_para_prever = [features_reais.values]

    # 4. Fazer a previsão com o modelo carregado
    previsao = modelo.predict(features_para_prever)
    resultado = "PHISHING (PERIGO!)" if previsao[0] == 0 else "LEGÍTIMO (SEGURO)"

    print("\n--- TESTE DE VERIFICAÇÃO ---")
    print(f"Rótulo Verdadeiro do Exemplo: {'LEGÍTIMO' if rotulo_verdadeiro == 1 else 'PHISHING'}")
    print(f"Previsão do Modelo:          {resultado}")

    if previsao[0] == rotulo_verdadeiro:
        print("\n✅ SUCESSO! O modelo previu corretamente o exemplo do dataset.")
    else:
        print("\n❌ FALHA! O modelo errou a previsão do exemplo do dataset.")

except FileNotFoundError:
    print("Erro: Arquivo do modelo ou do dataset não encontrado.")