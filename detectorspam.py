import pandas as pd #ler e organizar os dados em formato de tabela
from sklearn.model_selection import train_test_split #separa os dados em treino e teste
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.naive_bayes import MultinomialNB #algoritmo de classicação 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report #medem se o modelo está indo bem

# parte de carregar os dados
df = pd.read_csv("sms.tsv", sep="\t", header=None, names=["label", "message"])
## pd.read_csv carrega o arquivo numa tabela chamada de df, value_counts() mostra a quantiade de mensagens que existem de cada tipo
print(f"Total de mensagens: {len(df)}")
print(df["label"].value_counts())


#parte de preparar os dados
df["label_num"] = df["label"].map({"ham": 0, "spam": 1}) 

x = df["message"]
y = df["label_num"]

x_train, x_test, y_train, y_test = train_test_split( #train_test_split divide os dados
    x, y, test_size=0.2, random_state=42
)

#parte de transformar texto em números

vectorizer = CountVectorizer(stop_words="english") 
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)
#countvectorizer olha todas as mensagens de treino, monta um vocubulario com todas as palavras encontradas e transforma cada mensaghem num vetor que conta quantas vezes cada palavra do vocabulario aparece nela

#parte de treinar o modelo
model = MultinomialNB() #cria o modelo naive bayes 
model.fit(x_train_vec, y_train)

#parte de avaliar o modelo
y_pred = model.predict(x_test_vec)

print(f"Exatidão: {accuracy_score(y_test, y_pred):.2%}")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

#testar com nvoas mensagens
def prever(mensagem): 
    vec = vectorizer.transform([mensagem])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    resultado = "SPAM" if pred == 1 else "HAM (normal)"
    confianca = proba[pred]
    return resultado, confianca

mensagens_teste = [
    "Congrulations! You won a free prize, click here to claim now!",
    "Hey, are we still meeting for lunch tomorrow?",
]

for msg in mensagens_teste:
    resultado, confianca = prever(msg)
    print(f"Mensagens: {msg}")
    print(f"Classificação: {resultado} (confiaça: {confianca:.1%})")