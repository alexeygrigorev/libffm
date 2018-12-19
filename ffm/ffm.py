# coding: utf-8

import os
path = os.path.dirname(__file__)
lib_path = path + '/libffm.so'

# binding code

import numpy as np
import ctypes


class FFM_Parameter(ctypes.Structure):
    _fields_ = [
        ('eta', ctypes.c_float),
        ('lam', ctypes.c_float),
        ('nr_iters', ctypes.c_int),
        ("k", ctypes.c_int),
        ('normalization', ctypes.c_bool),
        ('auto_stop', ctypes.c_bool)
    ]


class FFM_Model(ctypes.Structure):
    _fields_ = [
        ("n", ctypes.c_int),
        ("m", ctypes.c_int),
        ("k", ctypes.c_int),
        ("W", ctypes.POINTER(ctypes.c_float)),
        ('normalization', ctypes.c_bool)
    ]


class FFM_Node(ctypes.Structure):
    _fields_ = [
        ("f", ctypes.c_int),
        ("j", ctypes.c_int),
        ("v", ctypes.c_float),
    ]


class FFM_Line(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.POINTER(FFM_Node)),
        ("label", ctypes.c_float),
        ("size", ctypes.c_int),
    ]


class FFM_Problem(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_int),
        ("num_nodes", ctypes.c_long),

        ("data", ctypes.POINTER(FFM_Node)),
        ("pos", ctypes.POINTER(ctypes.c_long)),
        ("labels", ctypes.POINTER(ctypes.c_float)),
        ("scales", ctypes.POINTER(ctypes.c_float)),

        ("n", ctypes.c_int),
        ("m", ctypes.c_int),
    ]

FFM_Node_ptr = ctypes.POINTER(FFM_Node)
FFM_Line_ptr = ctypes.POINTER(FFM_Line)
FFM_Model_ptr = ctypes.POINTER(FFM_Model)
FFM_Problem_ptr = ctypes.POINTER(FFM_Problem)

_lib = ctypes.cdll.LoadLibrary(lib_path)

_lib.ffm_convert_data.restype = FFM_Problem
_lib.ffm_convert_data.argtypes = [FFM_Line_ptr, ctypes.c_int]

_lib.ffm_init_model.restype = FFM_Model
_lib.ffm_init_model.argtypes = [FFM_Problem_ptr, FFM_Parameter]

_lib.ffm_train_iteration.restype = ctypes.c_float
_lib.ffm_train_iteration.argtypes = [FFM_Problem_ptr, FFM_Model_ptr, FFM_Parameter]

_lib.ffm_predict_array.argtypes = [FFM_Node_ptr, ctypes.c_int, FFM_Model_ptr]
_lib.ffm_predict_array.restype = ctypes.c_float

_lib.ffm_predict_batch.restype = ctypes.POINTER(ctypes.c_float)
_lib.ffm_predict_batch.argtypes = [FFM_Problem_ptr, FFM_Model_ptr]

_lib.ffm_load_model_c_string.restype = FFM_Model
_lib.ffm_load_model_c_string.argtypes = [ctypes.c_char_p]

_lib.ffm_save_model_c_string.argtypes = [FFM_Model_ptr, ctypes.c_char_p]

_lib.ffm_cleanup_data.argtypes = [FFM_Problem_ptr]
_lib.ffm_cleanup_prediction.argtypes = [ctypes.POINTER(ctypes.c_float)]

_lib.ffm_get_k_aligned.restype = ctypes.c_int
_lib.ffm_get_k_aligned.argtypes = [ctypes.c_int]

_lib.ffm_get_kALIGN.restype = ctypes.c_int

# some wrapping to make it easier to work with


def wrap_tuples(row):
    size = len(row)
    nodes_array = (FFM_Node * size)()

    for i, (f, j, v) in enumerate(row):
        node = nodes_array[i]
        node.f = f
        node.j = j
        node.v = v

    return nodes_array


def wrap_dataset_init(X, target):
    l = len(target)
    data = (FFM_Line * l)()

    for i, (x, y) in enumerate(zip(X, target)):
        d = data[i]
        nodes = wrap_tuples(x)
        d.data = nodes
        d.label = y
        d.size = nodes._length_

    return data


def wrap_dataset(X, y):
    line_array = wrap_dataset_init(X, y)
    return _lib.ffm_convert_data(line_array, line_array._length_)


class FFMData():

    def __init__(self, X=None, y=None):
        if X is not None and y is not None:
            self._data = wrap_dataset(X, y)
        else:
            self._data = None

    def __del__(self):
        _lib.ffm_cleanup_data(self._data)

    def num_rows(self):
        return self._data.size

# FFM model


class FFM():

    def __init__(self, eta=0.2, lam=0.00002, k=4):
        self._params = FFM_Parameter(eta=eta, lam=lam, k=k)
        self._model = None

    def read_model(self, path):
        path_char = ctypes.c_char_p(path.encode())
        model = _lib.ffm_load_model_c_string(path_char)
        self._model = model
        return self

    def save_model(self, path):
        model = self._model
        path_char = ctypes.c_char_p(path.encode())
        _lib.ffm_save_model_c_string(model, path_char)

    def init_model(self, ffm_data):
        params = self._params
        model = _lib.ffm_init_model(ffm_data._data, params)
        self._model = model
        return self

    def iteration(self, ffm_data):
        data = ffm_data._data
        model = self._model
        params = self._params
        loss = _lib.ffm_train_iteration(data, model, params)
        return loss

    def predict(self, ffm_data):
        data = ffm_data._data
        model = self._model

        pred_ptr = _lib.ffm_predict_batch(data, model)

        size = data.size
        pred_ptr_address = ctypes.addressof(pred_ptr.contents)
        array_cast = (ctypes.c_float * size).from_address(pred_ptr_address)

        pred = np.ctypeslib.as_array(array_cast)
        pred = np.copy(pred)
        _lib.ffm_cleanup_prediction(pred_ptr)
        return pred

    def _predict_row(self, nodes):
        n = nodes._length_
        model = self._model
        pred = _lib.ffm_predict_array(nodes, n, model)
        return pred

    def fit(self, X, y, num_iter=10):
        ffm_data = FFMData(X, y)
        self.init_model(ffm_data)

        for i in range(num_iter):
            self.model.iteration(ffm_data)

        return self
		
	def get_W(self):
		'''
		Returns the model vectors W of shape n x k, where n is the number of feature indexes.
		
		Given input X, the computation of probabilites can then be done in python as follows:
		
		import itertools
		i = 0 #the sample index for which we want to compute probabilities
		sig = lambda x: 1/(1+np.exp(-x)) #sigmoid function
		pairs = itertools.combinations(X[i],2)
		proba = sig(sum( (v0*W[i0,j1]) @ (v1*W[i1,j0]) for (j0,i0,v0), (j1,i1,v1) in pairs ))
		
		This shoud be the same as model.predict(X)[i]		
		'''
			m = self._model.m
		n = self._model.n
		k = self._model.k

		k_aligned = self._lib.ffm_get_k_aligned(k)
		kALIGN = self._lib.ffm_get_kALIGN()
		align0 = 2*k_aligned
		align1 = m*align0

		W = np.ctypeslib.as_array(self._model.W,(1,n*align1))[0]
		W = W.reshape((n,m,align0))
		I = np.arange(k) + ( np.floor(np.arange(align0)/kALIGN).astype(int)*kALIGN )[:k]
		W = W[:,:,I]
		return W


def read_model(path):
    return FFM().read_model(path)
