import numpy as np
from scipy.stats import ttest_rel

lr =  [1.4332299, 1.74749452, 0.40589238, 0.55372623, 0.5683    , 0.46214522, 1.69417236, 0.42056094, 0.73574744, 0.39635367]
ann = [ 1.25530129, 1.72391454, 0.34606335, 0.45478524, 0.49555388, 0.42671814
, 1.48504843, 0.49955183, 0.59647154, 0.36642598]

print "t-value and p-value"
print (ttest_rel(np.asarray(lr),np.asarray(ann)))
