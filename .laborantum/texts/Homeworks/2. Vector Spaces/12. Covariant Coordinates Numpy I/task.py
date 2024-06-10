#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

get_ipython().system('{sys.executable} -m pip -q install numpy json-tricks torch jupyter nbconvert')


# In[3]:


import json_tricks

path = Path('.laborantum/texts/Homeworks/2. Vector Spaces/12. Covariant Coordinates Numpy I')

debug_cases = json_tricks.load(
    str(path / 'testcases' / 'debug_cases.json'))
public_cases = json_tricks.load(
    str(path / 'testcases' / 'public_cases.json'))


# In[4]:


import numpy as np

def get_covariant_coordinates(B, x):
    # B * res = x
    res = np.dot(np.inverse(B), x)
    return res


# In[13]:


import time

start = time.time()

debug_result = [get_covariant_coordinates(**x) for x in debug_cases]
answer = [get_covariant_coordinates(**x) for x in public_cases]

print(time.time() - start, '<- Elapsed time')

