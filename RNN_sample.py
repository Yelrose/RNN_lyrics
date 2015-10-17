import numpy as np
import cPickle

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
        W = 5 / np.sum(W*W) * W
    return W

def sample_from(vec,topk,id2word,thre):
    tk = list(range(len(vec)))
    tk = sorted(tk,cmp=lambda x,y:cmp(vec[x],vec[y]),reverse=True)[:topk]
    tk = [wd for wd in tk if vec[wd] > thre]
    sumf = 0
    print ' '
    for i in tk:
        print id2word[i],'pro',vec[i]


        sumf += vec[i]
    p = np.random.uniform(low=0.0,high=sumf)
    sumf  = 0.
    for i in tk:
        sumf += vec[i]
        if sumf >= p: return i
    return tk[-1]




def sample(id2word,model,start,len_t,topk):
    #print X,y
    HH = model['HH']
    XH = model['XH']
    W = model['W']
    HO = model['HO']
    hs = model['hs']
    hx = model['hx']
    lh =  np.zeros(hs) #initialize first state
    word = []
    word.append(id2word[start])
    for i in xrange(len_t):
        x = W[start]
        lh = sigmoid(HH.dot(lh) + XH.dot(x))
        tt = HO.dot(lh)
        vec = softmax(tt)
        id = sample_from(vec,topk,id2word,0.002)
        start = id
        word.append(id2word[start])
    return word










if __name__ == '__main__':
    GECI = []
    fp = open('./geci.txt')
    dic = {}
    id2word = {}
    geci = None
    countff = 0
    while True:
        line = fp.readline()
        if not line: break
        if line[0:5] == 'start':
            if geci is not None:
                GECI.append(geci)
                countff += 1
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
        id2word[countf] = key
        countf += 1
    model = cPickle.load(open('./RNN_model_iter_77','rb'))
    while True:
        len_t = raw_input('length:')
        start = raw_input('start:')
        topk = raw_input('topk:')
        topk = int(topk)
        start = start.decode('utf-8')
        if start not in dic:
            continue
        start = dic[start]
        len_t = int(len_t)
        word = sample(id2word,model,start,len_t,topk)
        print ' '.join(word)

