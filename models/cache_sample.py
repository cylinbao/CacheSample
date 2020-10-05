import numpy as np
import scipy.sparse as sp
import time

def cache_sample_csr(csr, s_len):
    print("Processing cache sampling with length", s_len)
    t0 = time.time()
    nnode = csr.shape[0]
    row_ptr = csr.indptr
    col_ind = csr.indices

    _row_ptr = [0]
    _col_ind = []
    for i in range(nnode):
        nnz = row_ptr[i+1] - row_ptr[i]
        col = col_ind[row_ptr[i]:row_ptr[i+1]]
        if nnz > s_len:
            _col = col[:s_len]
        else:
            _col = col
        _col_ind.append(_col.astype(np.int32))
        _row_ptr.append(_row_ptr[-1] + _col.shape[0])

    _row_ptr = np.array(_row_ptr, dtype=np.int32)
    _col_ind = np.concatenate(_col_ind)

    _csr = sp.csr_matrix((np.ones(_col_ind.shape[0], dtype=np.float32),
           _col_ind, _row_ptr), shape=csr.shape)
    print("sampled nnz/nnz: {}/{}".format(_csr.nnz, csr.nnz))
    print("sample rate: {:.5f}%".format(_csr.nnz/csr.nnz*100))
    print("cache_sampel_csr takes {:.6f}s".format(time.time() - t0))
    return _csr

def cache_sample_rand_csr(csr, s_len):
    print("Processing cache sampling random with length", s_len)
    t0 = time.time()
    nnode = csr.shape[0]
    row_ptr = csr.indptr
    col_ind = csr.indices

    _row_ptr = [0]
    _col_ind = []
    for i in range(nnode):
        nnz = row_ptr[i+1] - row_ptr[i]
        col = col_ind[row_ptr[i]:row_ptr[i+1]]
        if nnz > s_len:
            idx = np.zeros(col.shape[0], dtype=np.bool)
            idx[:s_len] = True
            np.random.shuffle(idx)
            _col = col[idx]
        else:
            _col = col
        _col_ind.append(_col.astype(np.int32))
        _row_ptr.append(_row_ptr[-1] + _col.shape[0])

    _row_ptr = np.array(_row_ptr, dtype=np.int32)
    _col_ind = np.concatenate(_col_ind)

    _csr = sp.csr_matrix((np.ones(_col_ind.shape[0], dtype=np.float32),
            _col_ind, _row_ptr), shape=csr.shape)
    print("sampled nnz/nnz: {}/{}".format(_csr.nnz, csr.nnz))
    print("sampled rate: {:.5f}".format(_csr.nnz/csr.nnz))
    print("cache_sampel_rand_csr takes {:.6f}s".format(time.time() - t0))
    return _csr

def cache_sample_simrand_csr(csr, s_len):
    print("Processing cache sampling simulated random with length", s_len)
    t0 = time.time()
    nnode = csr.shape[0]
    row_ptr = csr.indptr
    col_ind = csr.indices

    _row_ptr = [0]
    _col_ind = []
    for i in range(nnode):
        lb = row_ptr[i]
        hb = row_ptr[i+1]
        nnz = hb - lb
        col = col_ind[lb:hb]
        if nnz < s_len:
            _col = col
        else:
            _col = []
            for ss in range(s_len):
                offset = ((ss*577) % nnz)
                _col.append(col[offset])
        _col = np.array(_col)
        _col_ind.append(_col.astype(np.int32))
        _row_ptr.append(_row_ptr[-1] + _col.shape[0])

    _row_ptr = np.array(_row_ptr, dtype=np.int32)
    _col_ind = np.concatenate(_col_ind)

    _csr = sp.csr_matrix((np.ones(_col_ind.shape[0], dtype=np.float32),
            _col_ind, _row_ptr), shape=csr.shape)
    print("sampled nnz/nnz: {}/{}".format(_csr.nnz, csr.nnz))
    print("sampled rate: {:.5f}".format(_csr.nnz/csr.nnz))
    print("cache_sample_simrand_csr takes {:.6f}s".format(time.time() - t0))
    return _csr

def cache_sample_simrand_csr_2(csr, s_len):
    print("Processing cache sampling simulated random with length", s_len)
    t0 = time.time()
    nnode = csr.shape[0]
    row_ptr = csr.indptr
    col_ind = csr.indices

    _row_ptr = [0]
    _col_ind = []
    for i in range(nnode):
        lb = row_ptr[i]
        hb = row_ptr[i+1]
        nnz = hb - lb
        col = col_ind[lb:hb]

        _col = []
        for ss in range(s_len):
            offset = ((ss*577) % nnz)
            if ss < nnz:
                _col.append(col[offset])
        _col = np.array(_col)
        _col_ind.append(_col.astype(np.int32))
        _row_ptr.append(_row_ptr[-1] + _col.shape[0])

    _row_ptr = np.array(_row_ptr, dtype=np.int32)
    _col_ind = np.concatenate(_col_ind)

    _csr = sp.csr_matrix((np.ones(_col_ind.shape[0], dtype=np.float32),
            _col_ind, _row_ptr), shape=csr.shape)
    print("sampled nnz/nnz: {}/{}".format(_csr.nnz, csr.nnz))
    print("sampled rate: {:.5f}".format(_csr.nnz/csr.nnz))
    print("cache_sample_simrand_csr takes {:.6f}s".format(time.time() - t0))
    return _csr

def cache_sample_uni_csr(csr, s_len):
    print("Processing cache sampling uniform with length", s_len)
    t0 = time.time()
    nnode = csr.shape[0]
    row_ptr = csr.indptr
    col_ind = csr.indices

    _row_ptr = [0]
    _col_ind = []
    for i in range(nnode):
        nnz = row_ptr[i+1] - row_ptr[i]
        col = col_ind[row_ptr[i]:row_ptr[i+1]]
        if nnz > s_len:
            _col = np.random.choice(col, s_len)
            # idx = np.zeros(col.shape[0], dtype=np.bool)
            # idx[:s_len] = True
            # np.random.shuffle(idx)
            # _col = col[idx]
        else:
            _col = col
        _col_ind.append(_col.astype(np.int32))
        _row_ptr.append(_row_ptr[-1] + _col.shape[0])

    _row_ptr = np.array(_row_ptr, dtype=np.int32)
    _col_ind = np.concatenate(_col_ind)

    _csr = sp.csr_matrix((np.ones(_col_ind.shape[0], dtype=np.float32),
            _col_ind, _row_ptr), shape=csr.shape)
    print("sampled nnz/nnz: {}/{}".format(_csr.nnz, csr.nnz))
    print("sampled rate: {:.5f}".format(_csr.nnz/csr.nnz))
    print("cache_sampel_rand_csr takes {:.6f}s".format(time.time() - t0))
    return _csr
