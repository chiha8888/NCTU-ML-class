import os
from math import factorial

if __name__=='__main__':
    path=input('path: ')
    name=input('name: ')
    fp=open(os.path.join(path,name),'r')
    a=int(input('init beta parameter a: '))
    b=int(input('init beta parameter b: '))
    line=fp.readline()
    case=1
    while line:
        positive=line.count('1')
        negative=line.count('0')
        total=positive+negative
        positiveProb=positive/total
        negativeProb=negative/total
        likelihood=factorial(total)/factorial(positive)/factorial(negative)*positiveProb**positive*negativeProb**negative
        print('case {}: {}'.format(case,line),end='')
        print('Likelihood: {}'.format(likelihood))
        print('Beta prior:     a={} b={}'.format(a,b))
        a+=positive
        b+=negative
        print('Beta posterior: a={} b={}'.format(a,b))
        print()
        line=fp.readline()
        case+=1