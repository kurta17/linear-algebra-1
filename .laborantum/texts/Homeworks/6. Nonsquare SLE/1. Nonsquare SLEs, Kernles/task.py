import numpy as np

answer = {
    "task_1": {
        "partial": np.array([[2], [1], [0]]).astype("int64"),
        "null_space":  np.array([[1], [0], [1]]).astype("int64"),
    },
    "task_2": {
        "partial": np.array([[6], [8], [0], [0], [0]]).astype("int64"),
        "null_space": [
            np.array([[-1], [-1], [1], [0], [0]]).astype("int64"),
            np.array([[-1], [-1], [0], [1], [0]]).astype("int64"),
            np.array([[-1], [-1], [0], [0], [1]]).astype("int64")
        ],
    },
    "task_3": {
        "partial": np.array([[3], [-11], [5], [0], [0]]).astype("int64"),
        "null_space": [
            np.array([[-1], [0], [-1], [1], [0], [0]]).astype("int64"),
            np.array([[-2], [0], [-1], [0], [1], [0]]).astype("int64"),
            np.array([[-3], [0], [-1], [0], [0], [1]]).astype("int64"),
        ],
    },
}