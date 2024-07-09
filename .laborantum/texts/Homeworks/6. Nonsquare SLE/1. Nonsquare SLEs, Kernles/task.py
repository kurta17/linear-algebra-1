import numpy as np

answer = {
    'task_1': {
        'partial': np.array([[ 1.],[-0.],[ 0.]]), 
       'null_space': [np.array([[-1.],
       [ 3.],
       [ 0.]]), 
       np.array([[-1.],[ 0.],[ 1.]])]
    },

    'task_2': {
        'partial': np.array([[0]]).T,
        'null_space': [
            np.array([[0]]).T,
            np.array([[0]]).T,
            np.array([[0]]).T
        ]
    },
    'task_3': {
        'partial': np.array([[0]]).T,
        'null_space': [
            np.array([[0]]).T,
            np.array([[0]]).T,
            np.array([[0]]).T
        ]
    }
}