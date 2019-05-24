import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

from mlp import MLP

def preProcessing_Data(path):
    fp = open(path, 'r')

    X = [] # features
    Y = [] # labels

    for i in fp:
        X.append(list(map(float, i.split()[:-10])))  #slicing inputs
        Y.append(list(map(float, i.split()[-10:])))  #slicing training classes

    fp.close()

    return (np.array(X, dtype=float),np.array(Y, dtype=int))

def main():
    (X, Y) = preProcessing_Data("./data/semeion.data")
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    sizeHidden = 30

    while(sizeHidden <= 100):
        mlp = MLP(X_train, Y_train, hL_size=sizeHidden, oL_size = 10)
        mlp.backpropagation()

        Y_pred = []
        
        for i in X_test:
            Y_pred.append(mlp.predict(i))
            
        score = accuracy_score(Y_test, np.array(Y_pred))
        
        print("Nº de neurônios: %d | Score: %f " % (sizeHidden, score))
    
        sizeHidden += 10
    

    #----------------------------------------------------

    eta_aux = 1e-4

    while(eta_aux <= 1):
        mlp = MLP(X_train, Y_train, hL_size=50, oL_size = 10, eta=eta_aux)
        mlp.backpropagation()

        Y_pred = []
        
        for i in X_test:
            Y_pred.append(mlp.predict(i))
            
        score = accuracy_score(Y_test, np.array(Y_pred))
        
        print("Eta: %f | Score: %f" % (eta_aux, score))
        
        eta_aux *= 1e+2

    #-----------------------------------------------------

    kf10 = KFold(n_splits=10)

    for train_idx, test_idx in kf10.split(X):
        X_train, Y_train = X[train_idx], Y[train_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        mlp = MLP(X_train, Y_train, hL_size=50, oL_size=10)
        mlp.backpropagation()
        
        Y_pred = []

        for i in X_test:
            Y_pred.append(mlp.predict(i))
    
        score = accuracy_score(Y_test, np.array(Y_pred))
        print("Score: ", score)


if __name__ == "__main__":
    main()