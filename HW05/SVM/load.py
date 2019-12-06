import numpy as np
import csv

def load_x(path):
    re=[]
    with open(path) as cvsfile:
        rows=csv.reader(cvsfile)
        for row in rows:
            re.append([float(v) for v in row])
    re=np.asarray(re,dtype='float')
    return re
def load_y(path):
    re=[]
    with open(path) as csvfile:
        rows=csv.reader(csvfile)
        for row in rows:
            re.append(int(row[0]))
    re=np.asarray(re,dtype=int)
    return re