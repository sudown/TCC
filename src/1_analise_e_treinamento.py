import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("--- INICIANDO PROCESSO DE ANÁLISE E TREINAMENTO ---")

# --- 1. CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
print("\n[ETAPA 1/4] Carregando e preparando os dados...")
df = pd.read_csv('./data/Phishing_Legitimate_full.csv')

X = df.drop(['CLASS_LABEL', 'id'], axis=1) 
y = df['CLASS_LABEL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Dados carregados e divididos com sucesso!")

# --- 2. TREINAMENTO E AVALIAÇÃO DO RANDOM FOREST ---
print("\n[ETAPA 2/4] Treinando e avaliando o modelo Random Forest...")
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_train, y_train)
previsoes_rf = modelo_rf.predict(X_test)
acuracia_rf = accuracy_score(y_test, previsoes_rf)

print("--- Resultados do Random Forest ---")
print(f"Acurácia: {acuracia_rf * 100:.2f}%")
print("Relatório de Classificação:")
print(classification_report(y_test, previsoes_rf))

# --- 3. TREINAMENTO E AVALIAÇÃO DO SVM ---
print("\n[ETAPA 3/4] Treinando e avaliando o modelo SVM...")
modelo_svm = SVC(random_state=42)
modelo_svm.fit(X_train, y_train)
previsoes_svm = modelo_svm.predict(X_test)
acuracia_svm = accuracy_score(y_test, previsoes_svm)

print("--- Resultados do SVM ---")
print(f"Acurácia: {acuracia_svm * 100:.2f}%")
print("Relatório de Classificação:")
print(classification_report(y_test, previsoes_svm))

# --- 4. COMPARAÇÃO E SALVAMENTO DO MELHOR MODELO ---
print("\n[ETAPA 4/4] Salvando o melhor modelo...")
if acuracia_rf > acuracia_svm:
    melhor_modelo = modelo_rf
    nome_modelo = "Random Forest"
else:
    melhor_modelo = modelo_svm
    nome_modelo = "SVM"

caminho_salvar = './models/modelo_phishing.joblib'
joblib.dump(melhor_modelo, caminho_salvar)

print(f"O modelo {nome_modelo} foi o melhor e foi salvo em '{caminho_salvar}'")
print("\n--- PROCESSO CONCLUÍDO ---")