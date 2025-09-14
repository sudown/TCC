import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Carregando o dataset (como você já fez)
df = pd.read_csv('./Phishing_Legitimate_full.csv') # Ajuste o nome se for diferente

# PASSO 2 CORRIGIDO: Analisando a Variável-Alvo
print("\n--- Balanceamento da Variável-Alvo ---")
# Trocamos 'Result' por 'CLASS_LABEL'
print(df['CLASS_LABEL'].value_counts())

# Gerando o gráfico com a coluna correta

# PASSO 3 CORRIGIDO: Análise de Correlação
print("\n--- Correlação das Features com a Coluna 'CLASS_LABEL' ---")
# Para uma análise de correlação mais limpa, vamos remover a coluna 'id' que não é uma feature
df_for_corr = df.drop('id', axis=1)
correlation_matrix = df_for_corr.corr()

# Trocamos 'Result' por 'CLASS_LABEL'
correlation_with_target = correlation_matrix['CLASS_LABEL'].sort_values(ascending=False)
print(correlation_with_target)


# PASSO 4 CORRIGIDO: Preparação Final para a Modelagem
print("\n--- Preparando os Dados para Modelagem ---")
# Separando as features (X) da variável-alvo (y)
# Removemos a coluna alvo 'CLASS_LABEL' e a coluna de identificação 'id'
X = df.drop(['CLASS_LABEL', 'id'], axis=1) 
# A nossa variável-alvo 'y' é a coluna 'CLASS_LABEL'
y = df['CLASS_LABEL']

# Dividindo os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamanho do conjunto de treino (X_train): {X_train.shape}")
print(f"Tamanho do conjunto de teste (X_test): {X_test.shape}")

# --- PASSO 5: TREINAMENTO E AVALIAÇÃO DO MODELO (RANDOM FOREST) ---

# 1. Importando o algoritmo que vamos usar
from sklearn.ensemble import RandomForestClassifier

# 2. Importando as ferramentas de avaliação
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print("\n--- Iniciando Treinamento do Modelo Random Forest ---")

# Criando o modelo "em branco"
# random_state=42 garante que o resultado seja o mesmo toda vez que rodarmos, para consistência.
modelo_rf = RandomForestClassifier(random_state=42)

# TREINANDO O MODELO (aqui a mágica acontece!)
# Ele vai aprender os padrões dos dados de treino.
modelo_rf.fit(X_train, y_train)

print("Modelo treinado com sucesso!")

print("\n--- Iniciando Avaliação do Modelo ---")

# Fazendo previsões com os dados de teste (que o modelo nunca viu)
previsoes_rf = modelo_rf.predict(X_test)

# --- AVALIANDO OS RESULTADOS ---

# 1. Acurácia: De todas as previsões, quantas o modelo acertou?
acuracia = accuracy_score(y_test, previsoes_rf)
print(f"Acurácia do modelo: {acuracia * 100:.2f}%")

# 2. Matriz de Confusão: Mostra onde o modelo acertou e errou, em detalhe.
print("\nMatriz de Confusão:")
# Formato:
# [[Verdadeiro Negativo, Falso Positivo],
#  [Falso Negativo,    Verdadeiro Positivo]]
print(confusion_matrix(y_test, previsoes_rf))

# 3. Relatório de Classificação: Um resumo completo com precisão, recall e f1-score.
print("\nRelatório de Classificação:")
print(classification_report(y_test, previsoes_rf, target_names=['Phishing (-1)', 'Legítimo (1)']))

# --- PASSO 6: TREINAMENTO E AVALIAÇÃO DO SEGUNDO MODELO (SVM) ---

# 1. Importando o algoritmo SVM
from sklearn.svm import SVC

print("\n\n--- Iniciando Treinamento do Modelo SVM (Support Vector Machine) ---")

# Criando o modelo SVM "em branco"
modelo_svm = SVC(random_state=42)

# TREINANDO O MODELO SVM
modelo_svm.fit(X_train, y_train)

print("Modelo SVM treinado com sucesso!")

print("\n--- Iniciando Avaliação do Modelo SVM ---")

# Fazendo previsões com os dados de teste
previsoes_svm = modelo_svm.predict(X_test)

# --- AVALIANDO OS RESULTADOS DO SVM ---

# 1. Acurácia
acuracia_svm = accuracy_score(y_test, previsoes_svm)
print(f"Acurácia do modelo SVM: {acuracia_svm * 100:.2f}%")

# 2. Matriz de Confusão
print("\nMatriz de Confusão (SVM):")
print(confusion_matrix(y_test, previsoes_svm))

# 3. Relatório de Classificação
print("\nRelatório de Classificação (SVM):")
print(classification_report(y_test, previsoes_svm, target_names=['Phishing (-1)', 'Legítimo (1)']))