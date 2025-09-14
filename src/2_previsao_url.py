import joblib
import numpy as np
import pandas as pd # Usado para obter a ordem das colunas
from urllib.parse import urlparse

def extrair_features(url, ordem_das_colunas):
    features = {}
    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname if parsed_url.hostname else ''
        path = parsed_url.path
    except Exception as e:
        print(f"Erro ao parsear a URL: {e}")
        hostname = ''
        path = ''

    # Features simples
    features['NumDots'] = url.count('.')
    features['SubdomainLevel'] = len(hostname.split('.')) - 2 if len(hostname.split('.')) > 2 else 0
    features['PathLevel'] = len(path.split('/')) - 1 if path else 0
    features['UrlLength'] = len(url)
    features['NumDash'] = url.count('-')
    features['NumDashInHostname'] = hostname.count('-')
    features['AtSymbol'] = 1 if '@' in url else 0
    features['TildeSymbol'] = 1 if '~' in url else 0
    features['NumUnderscore'] = url.count('_')
    features['NumPercent'] = url.count('%')
    features['NumQueryComponents'] = len(parsed_url.query.split('&')) if parsed_url.query else 0
    features['NumAmpersand'] = url.count('&')
    features['NumHash'] = url.count('#')
    features['NumNumericChars'] = sum(c.isdigit() for c in url)
    features['NoHttps'] = 1 if parsed_url.scheme != 'https' else 0
    features['HostnameLength'] = len(hostname)
    features['PathLength'] = len(path)
    features['QueryLength'] = len(parsed_url.query)

    # Preenchendo as features restantes com um valor padrão (0)
    for feature in ordem_das_colunas:
        if feature not in features:
            features[feature] = 0

    # Retorna os valores na ordem correta
    return np.array([features[feature] for feature in ordem_das_colunas])

# --- SCRIPT PRINCIPAL ---
print("--- CARREGANDO MODELO DE DETECÇÃO DE PHISHING ---")
try:
    # Carrega a ordem das colunas do dataset original
    df_original = pd.read_csv('./data/Phishing_Legitimate_full.csv')
    ordem_colunas = df_original.drop(['CLASS_LABEL', 'id'], axis=1).columns

    # Carrega o modelo treinado
    modelo = joblib.load('./models/modelo_phishing.joblib')
    print("Modelo carregado com sucesso!")

    while True:
        print("\nDigite uma URL para testar (ou 'sair' para fechar):")
        url_usuario = input("> ")

        if url_usuario.lower() == 'sair':
            break

        # 1. Extrair features
        features = extrair_features(url_usuario, ordem_colunas)

        # 2. Ajustar formato
        features = features.reshape(1, -1)

        # 3. Fazer previsão
        previsao = modelo.predict(features)

        # 4. Mostrar resultado
        resultado = "PHISHING (PERIGO!)" if previsao[0] == 0 else "LEGÍTIMO (SEGURO)"
        print(f"CLASSIFICAÇÃO: A URL é provavelmente {resultado}")

except FileNotFoundError:
    print("Erro: Arquivo do modelo ou do dataset não encontrado.")
    print("Certifique-se de executar primeiro o script '1_analise_e_treinamento.py'")