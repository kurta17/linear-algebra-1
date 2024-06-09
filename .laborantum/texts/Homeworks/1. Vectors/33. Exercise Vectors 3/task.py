#!/usr/bin/env python
# coding: utf-8

# In[3]:


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

get_ipython().system('{sys.executable} -m pip -q install --user numpy json-tricks torch jupyter nbconvert')


# In[4]:


import json_tricks

path = Path('.laborantum/texts/Homeworks/1. Vectors/33. Exercise Vectors 3')


# In[5]:


debug_cases = json_tricks.load(str(path / 'testcases' / 'debug_cases.json'))
public_cases = json_tricks.load(str(path / 'testcases' / 'public_cases.json'))


# In[15]:


import numpy as np
import numpy.typing as npt
from typing import Dict

def vector_operations(x, y):
    expression = x + y
    dot_prod = np.dot(x, y)
    length_a = np.linalg.norm(x)
    length_b = np.linalg.norm(y)
    angle = np.arccos(dot_prod / (length_a * length_b))
    dir_a = x / length_a
    dir_b = y / length_b
    a_proj_b = (dot_prod / length_a) * dir_a 
    b_proj_a = (dot_prod / length_b) * dir_b
    a_orth_b = x - a_proj_b
    b_orth_a = y - b_proj_a

    answer = {
        'expression': expression,
        'dot_prod': dot_prod,
        'length_a': length_a,
        'length_b': length_b,
        'angle': angle,
        'dir_a': dir_a,
        'dir_b': dir_b,
        'a_proj_b': a_proj_b,
        'b_proj_a': b_proj_a,
        'a_orth_b': a_orth_b,
        'b_orth_a': b_orth_a
    }

    return answer


# In[16]:


import time

start = time.time()

debug_result = [vector_operations(**x) for x in debug_cases]
answer = [vector_operations(**x) for x in public_cases]

print(time.time() - start, '<- Elapsed time')

