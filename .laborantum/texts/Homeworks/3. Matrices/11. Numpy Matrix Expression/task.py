#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import sys
import os
from pathlib import Path
import json_tricks

path = Path('.').resolve()

index = str(path).find('.laborantum')

if index > 0:
    path = str(path)[:index]

os.chdir(path)

path = Path('.laborantum/texts/Homeworks/3. Matrices/11. Numpy Matrix Expression')

debug_cases = json_tricks.load(
    str(path / 'testcases' / 'debug_cases.json'))
public_cases = json_tricks.load(
    str(path / 'testcases' / 'public_cases.json'))


# In[6]:


import numpy as np
from scipy.linalg import expm

def calculate_expression(A, B, C, x):
    B_plus_2C = B + 2 * C
    
    AT_B_plus_2C = np.dot(A.T, B_plus_2C)
    
    I = np.eye(AT_B_plus_2C.shape[0]) 
    matrix_to_exponentiate = AT_B_plus_2C + 3 * I
    
    matrix_exp = expm(matrix_to_exponentiate)
    
    result = np.dot(matrix_exp, x)

    return result


# In[7]:


import time

start = time.time()

debug_result = [formula(**x) for x in debug_cases]
answer = [formula(**x) for x in public_cases]

print(time.time() - start, '<- Elapsed time')


# In[ ]:




