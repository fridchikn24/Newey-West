import numpy as np
import pandas as pd


def sig1(e: pd.DataFrame):
    sig = pd.DataFrame()
    lentil = int((e.len)-1)
    for i in range(1,lentil):
        sig[i] = e[i] * e[i+1]
    siguno = sig.sum/(e.len-1)
    return np.diagflat(siguno)

class linear_model:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        self.b = np.linalg.solve(x.T@x,x.T@y)
        e = y-x@self.b
        self.vb = self.vcov_b(e)
        self.se = np.sqrt(np.diagonal(self.vb))
        self.t = self.b/self.se
    def vcov_b(self,e):
        x = self.x
        return e.var()*np.linalg.inv(x.T@x)
class NW(linear_model):
    def vcov_b(self,e):
        x = self.x
        sig_sq = np.diagflat(e.var)
        sigil = sig1(e)
        sig_top = np.concatenate(np.zeroes((1,sigil.shape[1]+1)),sigil,axis = 1)
        sig_bot= np.concatenate(np.zeroes((sigil.shape[0]+1,1)),sigil,axis=0)
        omega = sig_sq + sig_top + sig_bot
        bread1 = np.linalg.inv(x.T@x)@x.T
        bread2 = x@np.linalg.inv(x.T@x)
        sandwich = bread1@omega@bread2
        return sandwich

df = pd.read_csv("RidingMowers.csv")
lm = linear_model()

