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
    ## YOUR CODE HERE

    answer = {
        'expression': 1,
        'dot_prod': 1,
        'length_a': 1,
        'length_b': 1,
        'angle': 1,
        'dir_a': 1,
        'dir_b': 1,
        'a_proj_b': 1,
        'b_proj_a': 1,
        'a_orth_b': 1,
        'b_orth_a': 1
    }

    return answer


# In[16]:


import time

start = time.time()

debug_result = [vector_operations(**x) for x in debug_cases]
answer = [vector_operations(**x) for x in public_cases]

print(time.time() - start, '<- Elapsed time')

