# --- PASSO 7: USANDO O MODELO TREINADO PARA PREVER NOVAS URLS ---
import numpy as np
from urllib.parse import urlparse

# ATENÇÃO: Esta é uma função SIMPLIFICADA apenas para demonstração.
# Ela não extrai todas as 48 features, então a precisão pode não ser a mesma.
# Mas o PROCESSO é exatamente este.
def extrair_features(url):
    features = {}
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname if parsed_url.hostname else ''
    path = parsed_url.path

    # Features simples que podemos extrair aqui
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
    
    # Preenchendo as features restantes com um valor padrão (0), pois não as estamos calculando
    # Pegamos a lista de todas as features do seu conjunto de treino X_train
    todas_as_features = X_train.columns
    for feature in todas_as_features:
        if feature not in features:
            features[feature] = 0 # Usando 0 como padrão
            
    # Retorna os valores na ordem correta das colunas
    return np.array([features[feature] for feature in todas_as_features])

# --- Vamos testar! ---
urls_para_testar = [
    "https://www.google.com",
    "https://www.bancodobrasil.com.br/portalsite/inicio/",
    "http://login-itau-seguranca.com-br.info/cliente/update.html" # URL com cara de phishing
]

print("\n\n--- Testando o modelo com URLs novas ---")

for url in urls_para_testar:
    # 1. Extrair as features da URL
    features_da_url = extrair_features(url)
    
    # 2. O Scikit-learn espera um array 2D, então ajustamos o formato
    features_da_url = features_da_url.reshape(1, -1)
    
    # 3. Fazer a previsão com o melhor modelo (Random Forest)
    previsao = modelo_rf.predict(features_da_url)
    
    # 4. Mostrar o resultado de forma amigável
    resultado = "PHISHING" if previsao[0] == 0 else "LEGÍTIMO"
    print(f"A URL '{url}' foi classificada como: {resultado}")