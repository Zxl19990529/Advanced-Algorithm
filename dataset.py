import numpy as np

def data_iterator(x, y, batch_size, shuffle=True):
    indx = list(range(len(x)))
    if shuffle:
        np.random.shuffle(indx)

    for start_idx in range(0, len(x), batch_size):
        end_idx = min(start_idx + batch_size, len(x)) # use the tail of the dataset
        yield x[indx[start_idx: end_idx]], y[indx[start_idx: end_idx]]