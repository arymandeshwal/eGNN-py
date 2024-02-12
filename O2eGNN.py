import numpy as np

# def matlab_array():
#   data = {}

#   def set_value(index, value):
#     # Convert index to tuple-based indexing, starting from 1
#     key = tuple(i - 1 for i in index)
#     data[key] = value

#   def get_value(index):
#     # Convert index to tuple-based indexing, starting from 1
#     key = tuple(i - 1 for i in index)
#     return data.get(key)

#   def __str__():
#     # Build a string representation based on stored data
#     rows = max(k[0] for k in data) + 1
#     cols = max(k[1] for k in data) + 1
#     output = [[0] * cols for _ in range(rows)]
#     for key, value in data.items():
#       output[key[0]][key[1]] = value
#     return str(output)

#   return set_value, get_value, __str__

def insert_element_at_index(lst, index, value):
    if index > len(lst):
        lst += [0] * (index - len(lst))
    lst.insert(index, value)
    return lst


# Loading data
#Data = np.load('DataStream.npy')

#for testing purposes using original data
import scipy.io
Data = scipy.io.loadmat('../DataStream.mat')['Data']

X = Data[:, :2]  # Inputs
Y = Data[:, 2]   # Outputs
n = X.shape[1]   # No. of inputs
m = 1            # No. of outputs

# Parameters
Rho = 0.85        # Granularity
hr = 40           # No. of iterations to adapt the granules size / Inactivity: threshold for deletion
eta = 2
Beta = 0          # Weights (w) drop off constant
chi = 0.1         # Neutral elements (e) increasing constant
zeta = 0.9        # Weights (delta) adjusting constant
c = -1             # No. of granules
w = np.array([])  # Weights matrix - evolving layer
e = np.array([])  # Vector relat. to the nullneurons neutral element
delta = np.array([])  # Weights vector - aggregation layer
counter = 1
alpha = 0


# Normalization
Me = np.minmax(X, axis=0)
X = (X - Me[0]) / (Me[1] - Me[0])

# Simulating data stream
for h in range(X.shape[0]):
    # Choose data pairs sequentially
    x = X[h]
    y = Y[h]
    
    if h == 0:  # First step
        # Creating granule
        c += 1
        w = np.vstack([w, np.ones((1, n))])
        e = insert_element_at_index(e, 0)
        delta = insert_element_at_index(delta, 1)
        coef = insert_element_at_index([y])  # Class
        
        # Input granule
        l = x - Rho/2  # Lower bound
        lambda_ = x     # Intermediary point 1
        Lambda = x      # Intermediary point 2
        L = x + Rho/2   # Upper bound
        l = np.maximum(l, 0)
        L = np.minimum(L, 1)
        
        # Storing the example
        PointInG = np.zeros((1, 2011))
        PointInG[c-1, 0] = h
        Act = np.zeros((1,))
        
        # Test
        C = np.round(np.random.rand())  # Class - flip coin
        StoreNoRule = np.zeros((1,))    # No. of rules
        VecRho = np.array([Rho])        # Granule size along iterations
        
        # Desired output becomes available
        if y == C:
            Right = np.array([1])
            Wrong = np.array([0])
        else:
            Right = np.array([0])
            Wrong = np.array([1])
        
    else:  # Other steps
        # Test
        
        # Compute compatibility between x and granules
        Xnorm = np.zeros((c, n))
        for i in range(c):  # For all granules
            for j in range(n):  # For all features
                if x[j] >= l[i, j] and x[j] <= L[i, j]:  # Non-empty intersection
                    Xnorm[i, j] = 1 - (abs(x[j] - l[i, j]) + abs(x[j] - lambda_[i, j]) + abs(x[j] - Lambda[i, j]) + abs(x[j] - L[i, j])) / 4
                else:
                    Xnorm[i, j] = 0
        
        # Aggregation layer weights
        Xnorm *= w
        
        # T-S neuron: min
        o = np.min(Xnorm, axis=1)
        
        # Output layer weights
        o *= delta
        
        # T-S neuron: max
        I = np.argmax(o)  # Granule I is the closest
        
        # Vector for plot ROC (out)
        oClass1 = 0
        oClass2 = 0
        for i in range(coef.size):
            if coef[i] == 0:
                oClass1 = max(oClass1, o[i])
            elif coef[i] == 1:
                oClass2 = max(oClass2, o[i])
        out = np.array([oClass1 / (oClass1 + oClass2), oClass2 / (oClass1 + oClass2)])
        
        # Prediction error - vectors for plot
        C = coef[I]  # Class predicted
        StoreNoRule = np.array([c])  # No. of rules
        
        # Desired output becomes available
        if y == C:
            Right = np.append(Right, 1)
            Wrong = np.append(Wrong, 0)
        else:
            Right = np.append(Right, 0)
            Wrong = np.append(Wrong, 1)
        
        # Train
        
        # Rules that can fit x
        I = np.where(o > 0)[0]
        I = I[coef[I] == y]
        
        Flag = False
        if I.size == 0:
            Flag = True  # No rule encloses x
        
        if Flag:  # Case 0: No granule accommodates x
            # Creating granule
            c += 1
            w = np.vstack([w, np.ones((1, n))])
            e = np.append(e, 0)
            delta = np.append(delta, 1)
            coef = np.append(coef, y)  # Class
            # Input granule
            l = np.vstack([l, x - Rho/2])      # Lower bound
            lambda_ = np.vstack([lambda_, x])  # Intermediary point 1
            Lambda = np.vstack([Lambda, x])    # Intermediary point 2
            L = np.vstack([L, x + Rho/2])      # Upper bound
            # Storing the example
            PointInG = np.vstack([PointInG, np.zeros((1, 2011))])
            PointInG[-1, 0] = h
            Act = np.append(Act, 0)
        else:  # Adaptation of the most qualified granule
            # Case >2: more than one rule fits x
            if I.size >= 2:
                pos = np.argmax(o[I])
                I = np.array([I[pos]])  # Granule I should be used for training
            # Adapting granule I - antecedent part
            for j in range(n):
                if x[j] > ((lambda_[I, j] + Lambda[I, j]) / 2) - Rho/2 and x[j] < lambda_[I, j]:
                    lambda_[I, j] = x[j]  # Core expansion
                if x[j] > lambda_[I, j] and x[j] < ((lambda_[I, j] + Lambda[I, j]) / 2):
                    lambda_[I, j] = x[j]  # Core contraction
                if x[j] > ((lambda_[I, j] + Lambda[I, j]) / 2) and x[j] < Lambda[I, j]:
                    Lambda[I, j] = x[j]  # Core contraction
                if x[j] > Lambda[I, j] and x[j] < ((lambda_[I, j] + Lambda[I, j]) / 2) + Rho/2:
                    Lambda[I, j] = x[j]  # Core expansion
