import math
import numpy as np

# Initializes the corellaction weights matrix W 
# as the inner dot product of X and Y pattern maps
def learn(x,y):
  return x.T.dot(y)

# Initializes the corellaction weights matrix W as
# the sum of outer Kronecker products of corresponding
# patterns x and y from the maps X,Y, respectively.
def learn_op(x,y):
  return np.sum([np.outer(x,y) for x,y in zip(x,y)],axis=0)

# Bipolar threshold activation function
def bipolar_th(x):
    return 1 if x >= 0 else -1
# Applies the bipolar_th(x) function to the sum 
# of weighted inputs of all BAM's memory cells.
def activate(x):
    return np.vectorize(bipolar_th)(x)

# Recalls an association Y for the input pattern X, bidirectionally:
def recall(w,x,d='out'):
    end_of_recall = False; \
        y_pred = None; x_eval = y_pred
    # Compute the BAM output until the existing inputs x
    # are not equal to the new inputs x_eval (x != x_eval)
    while end_of_recall == False:
        # Compute the output y_pred of all memory cells, activated
        # by the bipolar threshold function F(X): [ w^T*x - forward, w*x - backwards ]
        y_pred = activate(w.T.dot(x) \
            if d == 'out' else w.dot(x))
        # Compute the new inputs x_eval for the next iteration:
        # [ w*y - forward, w^T*y - backwards]
        x_eval = activate(w.dot(y_pred) \
            if d == 'out' else w.T.dot(y_pred))
        # Check if x and x_eval are not the same. 
        # If not, assign the new inputs x_eval to x
        x,end_of_recall = x_eval,np.all(np.equal(x,x_eval))

    return y_pred  # Return the output pattern Y, recalled from the BAM.

# The BAM model of 8*10^3 inputs, 5*10^3 memory cells, with memory capacity - 20 patterns

patterns = 20; neurons = 8000; mm_cells = 5500

# Generate input (X) and output (Y) patterns maps of shapes (patterns x neurons) and (patterns by mm_cells)
X = np.array([1 if x > 0.5 else -1 for x in np.random.rand(patterns*neurons)],dtype=np.int8)

# Orthogonalize the input patterns (X) into the corresponding output patterns (Y) 
Y = np.array(-X[:patterns*mm_cells],dtype=np.int8)

# Reshape patterns into the input and output 2D-pattern maps X and Y
X = np.reshape(X,(patterns,neurons))
Y = np.reshape(Y,(patterns,mm_cells))

# Learn the BAM model with the associations of the input and output patterns X and Y
W = learn_op(X,Y) # W - the correlation weights matrix (i.e., the BAM's memory storage space)

print("Recalling the associations (Y) for the input patterns (X):\n")

# Recall an association (Y) for each input (X) and target output (Y') patterns, from X,Y
for x,y in zip(X,Y):
    y_pred = recall(W,x,'out') # y_pred - the predicted pattern Y
    # Check if the target and predicted patterns (Y) are identical, and display the results
    print("x =",x,"target =",y,"y =",-y_pred," :",np.any(-y_pred != y))

print("\r\nRecalling the associations (X) for the output patterns (Y):\n")

# Recall an association (X) for each output (Y) and target input (X) patterns, from X,Y
for x,y in zip(X,Y):
    x_pred = recall(W,y,d='in') # x_pred - the predicted pattern X
    # Check if the target and predicted patterns (X) are identical, and display the results
    print("y =",y,"target =",x,"x =",-x_pred," :",np.any(-x_pred != x))

# Distorts an input pattern map X
def poison(x,ratio=0.33,distort='yes'):
    p_fn = [ lambda x: 0 if np.random.rand() > 0.5 else x,
             lambda x: 1 if np.random.rand() > 0.5 else -1, ]

    x_shape = np.shape(x); x = np.reshape(x,-1)
    return np.reshape(np.vectorize(p_fn[distort == 'yes'])(x),x_shape)

# Predicting a randomly distorted pattern
print("\r\nPredicting a randomly distorted pattern X:\r\n")

# Select a pattern from X, randomly
pattern_n = np.random.randint(0,np.size(X,axis=0))

# Distort the input pattern with random 1's or -1's
x_dist = poison(X[pattern_n],distort='yes')

# Predict a correct association for the random pattern X
y_pred = recall(W,x_dist)

# Display the results
print("Output:\r\n")
print("x =",x,"target =",y,"y =",y_pred,":",np.any(y[pattern_n] != y_pred),"\r\n")
