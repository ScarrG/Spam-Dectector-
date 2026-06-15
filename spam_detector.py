"""
Detector de Spam - Projeto de IA para iniciantes
--------------------------------------------------
Este projeto usa Machine Learning para classificar mensagens de SMS
como "spam" ou "ham" (mensagem normal).

Conceitos abordados:
1. Carregar e explorar dados
2. Transformar texto em números (vetorização)
3. Treinar um modelo de classificação (Naive Bayes)
4. Avaliar a performance do modelo
5. Usar o modelo para prever novas mensagens
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# 1. CARREGAR OS DADOS
# O arquivo sms.tsv tem duas colunas separadas por TAB: rótulo (ham/spam) e a mensagem
print("Carregando dados...")
df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "message"])

print(f"Total de mensagens: {len(df)}")
print(df["label"].value_counts())
print()


# 2. PREPARAR OS DADOS
# Convertendo o rótulo em número: ham = 0, spam = 1
df["label_num"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]      # as mensagens (entrada)
y = df["label_num"]    # os rótulos (saída esperada)

# Dividindo em dados de treino (80%) e teste (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 3. TRANSFORMAR TEXTO EM NÚMEROS (vetorização)
# Modelos de ML não entendem texto, só números.
# O CountVectorizer transforma cada mensagem em um vetor que conta
# quantas vezes cada palavra aparece (Bag of Words).
print("Vetorizando texto...")
vectorizer = CountVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# 4. TREINAR O MODELO
# Naive Bayes é um algoritmo simples e muito usado para classificação de texto
print("Treinando modelo...")
model = MultinomialNB()
model.fit(X_train_vec, y_train)


# 5. AVALIAR O MODELO
y_pred = model.predict(X_test_vec)

print()
print("=== RESULTADOS ===")
print(f"Acurácia: {accuracy_score(y_test, y_pred):.2%}")
print()
print("Matriz de confusão:")
print(confusion_matrix(y_test, y_pred))
print()
print("Relatório de classificação:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))


# 6. TESTAR COM MENSAGENS NOVAS
def prever(mensagem):
    vec = vectorizer.transform([mensagem])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    resultado = "SPAM" if pred == 1 else "HAM (normal)"
    confianca = proba[pred]
    return resultado, confianca


print("=== TESTANDO COM MENSAGENS NOVAS ===")
mensagens_teste = [
    "Congratulations! You won a free prize, click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
    "URGENT: Your account has been suspended, verify now to avoid charges",
    "Can you send me the report when you get a chance?",
]

for msg in mensagens_teste:
    resultado, confianca = prever(msg)
    print(f"\nMensagem: {msg}")
    print(f"Classificação: {resultado} (confiança: {confianca:.1%})")
