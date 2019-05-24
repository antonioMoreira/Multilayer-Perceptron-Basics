import numpy as np
np.random.seed(1)

class MLP:
    def sigmoid(self, x, derivate=False):
        return 1/(1+np.exp(-x)) if not(derivate) else (x*(1-x))

    def init_Weights(self, row, col):
        # Gera valores aleatórios com distribuição normal padronizada (i.e., com média 0 e var 1)
        return (np.random.standard_normal(size=(row, col))*1e-4)
    
    def feedForward(self, weight, X):
        if(np.shape(weight)[1]-1 != len(X)):
            print("Feed Forward Error")
            return -1

        X = np.append(X, 1)
        net = np.dot(weight, np.transpose(X))
        
        return self.activationFunc(self, net)

    # Mean Squared Error
    def MSE(self, y, f_net, derivate=False):
        return sum(pow(y-f_net, 2)) if not(derivate) else (y-f_net)

    def backpropagation(self):
        squaredError = 2*self.threshold
        epoch = 0
        
        while(squaredError > self.threshold and epoch < self.n_Epochs):
            squaredError = 0 # ←←←←← tirar isso pra ver se converge
            
            row = 0
            for i in self.X:
                # » Forward pass
                f_net_hL = self.feedForward(self.hL_Weights, i)         # Feed Hidden Layer
                f_net_oL = self.feedForward(self.oL_Weights, f_net_hL)  # Feed Output Layer

                # Erro
                squaredError += self.MSE(self.Y[row], f_net_oL) # erro(esperado, obtido)
                d_squaredError = self.MSE(self.Y[row], f_net_oL, True) #derivada do MSE
                
                # » Backward pass
                old_oL_Weights = self.oL_Weights # para usar no ajuste da hidden layer
                
                # → Ajustando oL_Weights
                localGrd = np.multiply(d_squaredError, self.activationFunc(self, f_net_oL, True))
                self.oL_Weights += self.eta * np.multiply(np.transpose([localGrd]), np.append(f_net_hL, 1)) 

                # → Ajustando hL_Weights:
                localGrd = (localGrd @ old_oL_Weights[...,:-1]) * self.activationFunc(self, f_net_hL, True)
                self.hL_Weights += self.eta * np.transpose([localGrd]) * np.append(i, 1) 

                row += 1
            
            squaredError /= np.shape(self.X)[0] # MSE = MSE/N
            #print("MSE:\t", squaredError)
            epoch += 1

    def predict(self, sample):
        f_net_hL = self.feedForward(self.hL_Weights, sample)
        f_net_oL = self.feedForward(self.oL_Weights, f_net_hL)

        maximum = np.argmax(f_net_oL)

        for i in range(len(f_net_oL)):
            if (i != maximum):
                f_net_oL[i] = 0
            else:
                f_net_oL[i] = 1

        return np.array(f_net_oL, dtype=int)


    def __init__(self, X, Y, hL_size = 2, oL_size = 1, activationFunc = sigmoid, eta = 1e-2, 
                 n_Epochs = 5e2, threshold = 1e-2):
        self.iL_size = np.shape(X)[1]
        self.hL_size = hL_size
        self.oL_size = oL_size
        self.activationFunc = activationFunc
        self.eta = eta
        self.n_Epochs = n_Epochs
        self.threshold = threshold
        
        if(np.shape(X)[0] == np.shape(Y)[0]):
            (self.X, self.Y) = (X, Y)
        else:
            print("X and Y must be same row lenght")
        
        self.hL_Weights = self.init_Weights(self.hL_size, self.iL_size+1)
        self.oL_Weights = self.init_Weights(self.oL_size, self.hL_size+1)
        