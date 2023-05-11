# 손실함수

'''

def compute_cost(X, T, W, lambda_):

    epsilon = 1e-5
    
    N = len(T)
    
    K = np.size(T, 1)

    # cross-entropy(sum x)
    
    cost = - (1/N) * np.ones((1,N)) @ (np.multiply(np.log(softmax(X, W) + epsilon), T)) @ np.ones((K,1))

    # L2
    
    L2_cost = 0.5 * lambda_ * np.sum(W**2).reshape(1,1)
    
    return cost , L2_cost
    
 '''
 
 # w업데이트
 
크로스 엔트로피 + L2 penalty 미분

'''

 W = W - (learning_rate/batch_size) * (X_batch.T @ (softmax(X_batch, W) - T_batch) + lambda_*W)
 
'''

