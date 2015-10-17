import numpy as np
import sys
#some useful function
def sigmoid(x):
    return 1. /(1 + np.exp(-x))

def sigmoid_grad(x):
    return x * (1 -  x)

def relu(x):
    return np.maximum(0,x)

def softmax(x):
    x  = x  - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def clip_gradient(W):
    if np.sqrt(np.sum(W*W)) > 5:
        W = 5 / np.sqrt(np.sum(W*W)) * W
    return W
#solving per case
def recurrent_neural_network(X,Y,model,alpha):
    #print X,y
    HH = model['HH']
    XH = model['XH']
    W = model['W']
    HO = model['HO']
    hs = model['hs']
    hx = model['hx']
    ho = model['ho']

    #forward for the recurrent nerual network
    sz = len(X)
    s = np.zeros((sz+1,hs))
    grad_s = np.zeros((sz + 1,hs))
    x = np.zeros((sz+1,hx))
    grad_o = np.zeros((sz + 1,ho))
    o = np.zeros((sz+1,ho))

    s[0] = np.zeros(hs) #initialize first state
    loss = 0
    for i in xrange(sz): #forward
        x[i + 1] = W[X[i]]
        s[i + 1] = sigmoid(HH.dot(s[i]) + XH.dot(x[i + 1]))
        o[i + 1] = HO.dot(s[i + 1])
        loss += np.log((softmax(o[i+1])[Y[i]]))
        grad_o[i + 1] = softmax(o[i + 1])
        grad_o[i + 1][Y[i]] -= 1
        if (i+1) % 10 == 0 or (i == sz - 1):        #truncated
            j = i + 1
            grad_W = np.zeros(W.shape)
            grad_HH = np.zeros(HH.shape)
            grad_XH = np.zeros(XH.shape)
            grad_HO = np.zeros(HO.shape)
            while j > i + 1 - 10 and j > 0:
                grad_s[j] = HO.T.dot(grad_o[j])
                grad_HO += np.outer(grad_o[j],s[j])
                if j < i + 1:
                    grad_s[j] += HH.T.dot(grad_s[j+1]* (1 - s[j+1]) * (s[j+1]))
                    grad_HH += np.outer(grad_s[j+1]*(1-s[j+1])*s[j+1],s[j])
                grad_x = XH.T.dot(grad_s[j] * s[j] * (1 - s[j]))
                grad_XH += np.outer(grad_s[j]* s[j]* (1-s[j]),x[j])
                grad_W[X[j-1]] = grad_x
                j -= 1
            W += -alpha * clip_gradient(grad_W)
            HH += -alpha * clip_gradient(grad_HH)
            XH += -alpha * clip_gradient(grad_XH)
            HO += -alpha * clip_gradient(grad_HO)
    loc_loss = np.power(2,-loss/len(X))
    gobal_loss = loss
    return model,loc_loss,gobal_loss,len(X)
import random

def weight_initial(size):
    num_node = 1
    for i in size:
        num_node *= i
    return np.random.uniform(-np.sqrt(6)/num_node,np.sqrt(6)/num_node,size)


def training_neural_network(XX,YY,num_word,word_dim,max_iter,alpha,min_alpha,hidden_layer):
    model = {}
    model['hs'] = hidden_layer
    model['hx'] = word_dim
    model['ho'] = num_word
    model['W'] = np.random.uniform(-np.sqrt(6)/word_dim,np.sqrt(6)/word_dim,(num_word,word_dim))
    model['HH'] = weight_initial((hidden_layer,hidden_layer))
    model['XH'] = weight_initial((hidden_layer,word_dim))
    model['HO'] = weight_initial((num_word,hidden_layer))
    iter = 0
    all_word_count = 0
    for _ in XX:
        for __ in _:
            all_word_count += 1
    all_word_count *= max_iter
    while iter < max_iter:
        train = range(len(XX))
        random.shuffle(train)
        tot_loss = 0
        tot_word_count = 0
        nalpha = alpha * ( 1 - tot_word_count/all_word_count )
        nalpha = max(min_alpha,nalpha)
        for tn in train:
            model,loc_loss,loss,word_count = recurrent_neural_network(XX[tn],YY[tn],model,nalpha)
            tot_loss += loss
            tot_word_count += word_count
            print loc_loss
            sys.stdout.flush()

        print 'report iter',iter,'loss',np.power(2,(-tot_loss )/(tot_word_count))
        sys.stdout.flush()
        cPickle.dump(model,open('RNN_model_iter_%s' % str(iter),'w'))
        iter += 1
    return model












def make_X_y(GECI,dic):
    X = []
    y = []
    for geci in GECI:
        x = []
        for wd in geci:
            x.append(dic[wd])
        X.append(x[:-1])
        y.append(x[1:])
    return X,y

import cPickle

if __name__ == '__main__':
    GECI = []
    fp = open('./geci.txt')
    dic = {}
    geci = None
    countff = 0
    while True:
        line = fp.readline()
        if not line: break
        if line[0:5] == 'start':
            if geci is not None:
                GECI.append(geci)
                countff += 1
                #if countff == 5: break
            geci = []
            continue
        w = list(line.strip().decode("utf-8"))
        w.append(u"<\s>")
        for wd in w:
            dic[wd] = 1
        geci = geci + w
    countf = 0
    for key in dic:
        dic[key] = countf
        countf += 1
    X,Y = make_X_y(GECI,dic)
    print 'train #word',len(dic)
    sys.stdout.flush()

    model = training_neural_network(X,Y,len(dic),200,1000,alpha=0.5,min_alpha=0.05,hidden_layer=400)
    cPickle.dump(model,open('RNN_model','w'))





