import numpy as np

def cfor(first,test,update):
    while test(first):
        yield first
        first = update(first)

def WriteAprioriFile(X,titles=None,filename="AprioriFile.txt"):
    #Setup
    N, M = np.shape(X);
    f = open(filename, 'w');

    #Go through conversion loop
    if titles==None:
        for i in cfor(0,lambda i:i<N,lambda i:i+1):
            output = u"";
            for j in cfor(0,lambda j:j<M,lambda j:j+1):
                if (X[i,j] == 1):
                    output = output + str(j) + u",";
            f.write(output[:-1] + u"\n")
    else:
        for i in cfor(0,lambda i:i<N,lambda i:i+1):
            output = u"";
            for j in cfor(0,lambda j:j<M,lambda j:j+1):
                if (X[i,j] == 1):
                    output = output + titles[j] + u",";
            f.write(output[:-1] + u"\n")

    f.close;

