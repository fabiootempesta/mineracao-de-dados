#imports
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import neural_network
from sklearn import ensemble
from sklearn import linear_model
import pandas as pd



#Teste dos classificadores
def classify(classifier):
    runtime = []
    score = []
    
    
    #base original
    for i in range(15):
        start = time.time()
        classifier.fit(X_train, y_train)
        end = time.time()
        runtime.append(end-start)
        score.append(classifier.score(X_test, y_test))
    print((sum(score)/len(score)), (sum(runtime)/len(runtime)))
    runtime.clear()
    score.clear()
    
    
    #base preparada 1
    for i in range(15):
        start = time.time()
        classifier.fit(X_train, y_train)
        end = time.time()
        runtime.append(end-start)
        score.append(classifier.score(X1_test, y1_test))
    print((sum(score)/len(score)), (sum(runtime)/len(runtime)))
    runtime.clear()
    score.clear()
    
    
    #base preparada 2
    for i in range(15):
        start = time.time()
        classifier.fit(X_train, y_train)
        end = time.time()
        runtime.append(end-start)
        score.append(classifier.score(X2_test, y2_test))
    print((sum(score)/len(score)), (sum(runtime)/len(runtime)))
    runtime.clear()
    score.clear()
    
    
    #base preparada 3
    for i in range(15):
        start = time.time()
        classifier.fit(X_train, y_train)
        end = time.time()
        runtime.append(end-start)
        score.append(classifier.score(X3_test, y3_test))
    print((sum(score)/len(score)), (sum(runtime)/len(runtime)))



#base original
df = pd.read_csv('classifica.csv')
df = df.drop(columns=['ID'])

X_train, X_test, y_train, y_test = train_test_split(df.drop(columns = ['CLASS']), df['CLASS'], test_size = 0.1, random_state=5)

print(df)



#Preparação 1
df_p1 = df.copy()

df_p1['A2'] = df_p1['A2']/10
df_p1['A3'] = df_p1['A3']/100

X1_train, X1_test, y1_train, y1_test = train_test_split(df.drop(columns = ['CLASS']), df['CLASS'], test_size = 0.1, random_state=5)

print (df_p1)



#Preparação 2
df_p2 = df.copy()

df_p2 = df_p2.drop(columns=['A4','A6'])

X2_train, X2_test, y2_train, y2_test = train_test_split(df.drop(columns = ['CLASS']), df['CLASS'], test_size = 0.1, random_state=5)

print (df_p2)



#Preparação 3
df_p3 = df.copy()

df_p3 = df_p3.drop(columns=['A4','A6'])
df_p3['A2'] = df_p3['A2']/10
df_p3['A3'] = df_p3['A3']/100

X3_train, X3_test, y3_train, y3_test = train_test_split(df.drop(columns = ['CLASS']), df['CLASS'], test_size = 0.1, random_state=5)

print(df_p3)



#Perceptron Multicamadas

perceptron = neural_network.MLPClassifier(max_iter=12000, hidden_layer_sizes=50, solver = 'lbfgs')
classify(perceptron)


#Floresta Aleatória

rf = ensemble.RandomForestClassifier(n_estimators = 25, criterion= 'gini', max_features=None)
classify(rf)


#Gradiente Descendente Estocástico

sgd = linear_model.SGDClassifier(loss = 'modified_huber', penalty='elasticnet', max_iter = 12000)
classify(sgd)