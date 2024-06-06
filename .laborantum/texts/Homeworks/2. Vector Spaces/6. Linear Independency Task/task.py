#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
import os
from pathlib import Path

path = Path('.').resolve()

index = str(path).find('.laborantum')

if index > 0:
    path = str(path)[:index]

os.chdir(path)

get_ipython().system('{sys.executable} -m pip -q install  numpy json-tricks torch jupyter nbconvert')


# In[4]:


import json_tricks

path = Path('.laborantum/texts/Homeworks/2. Vector Spaces/6. Linear Independency Task')

debug_cases = json_tricks.load(
    str(path / 'testcases' / 'debug_cases.json'))
public_cases = json_tricks.load(
    str(path / 'testcases' / 'public_cases.json'))


# In[3]:


import numpy as np

def is_independent(A):
    A = A.astype('float64')  
    rows, cols = A.shape
    ans = True
    for col in range(cols):
        if A[col, col] == 0:
            for row in range(col+1, rows):
                if A[row, col] != 0:
                    A[[col, row]] = A[[row, col]]
                    break
        for row in range(col+1, rows):
            r = A[row, col] / A[col, col]
            A[row, :] -= r * A[col, :]
            if np.all(A[row, :] == 0):
                ans = False 
                break
            
    return ans


# In[4]:


import time

start = time.time()

debug_result = [is_independent(**x) for x in debug_cases]
answer = [is_independent(**x) for x in public_cases]

print(time.time() - start, '<- Elapsed time')

