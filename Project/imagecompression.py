#from telnetlib import X3PAD
from sys import set_coroutine_origin_tracking_depth
import numpy as np
import matplotlib.pyplot as plt
from TestQR import *
import matplotlib.image as img
from PIL import Image


filename = "big_tree.jpeg"

image = Image.open(filename)
mat = np.array(image)

#convert to grayscale if colored photo
#image_matrix = mat[:,:,1]/3.0 + mat[:,:,2]/3.0 + (mat[:,:,3]/3.0)
#print(mat)
#print(mat.shape)

test = np.random.rand(4,2)

#compute the factorization
U, S, Vt = np.linalg.svd(test)

S_keep = np.array(S.size())
epsilon = 3

#Only want to keep the largest singular values, larger than a given epsilon
for a in S:
    if a > epsilon:
        S_keep = np.insert(a)
        
S_full = np.diag(S_keep) # converts to diagonal matrix (the above function returns a 1D Array)

k = S_keep.size() # test


U_trunc = U[:,:k]
S_full_trunc = S_full[:k, :k]
Vt_trunc = Vt[:k, :]

Q, R = np.linalg.qr(test)



    
