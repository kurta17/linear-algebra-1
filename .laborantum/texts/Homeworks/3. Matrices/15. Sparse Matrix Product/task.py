#!/usr/bin/env python
# coding: utf-8

# In[2]:


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

path = Path(".laborantum/texts/Homeworks/3. Matrices/15. Sparse Matrix Product")

debug_cases = json_tricks.load(
    str(path / 'testcases' / 'debug_cases.json'))
public_cases = json_tricks.load(
    str(path / 'testcases' / 'public_cases.json'))


# In[3]:


import numpy as np

def sparseProduct(A, B):
    n, m = A.shape
    C = np.zeros((n, m))
    for r, c, v in B:
        C[:, r] += A[:, c] * v

    return np.array(C)


# In[ ]:


import time

start = time.time()

debug_result = [sparseProduct(**x) for x in debug_cases]
answer = [sparseProduct(**x) for x in public_cases]

print(answer)

print(time.time() - start, '<- Elapsed time')

