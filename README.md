## Python wrapper for libffm

This is a python wrapped for LibFFM library writen in C++

Installing it:

	python setup.py install  # or python setup.py develop


Using it:

	import ffm
	from sklearn.metrics import roc_auc_score

	# prepare the data
	# (field, index, value) format

	X = [[(1, 2, 1), (2, 3, 1), (3, 5, 1)],
	     [(1, 0, 1), (2, 3, 1), (3, 7, 1)],
	     [(1, 1, 1), (2, 3, 1), (3, 7, 1), (3, 9, 1)],]

	y = [1, 1, 0]

	ffm_data = ffm.FFMData(X, y)


	# train the model for 10 iterations

	n_iter = 10

	model = ffm.FFM(eta=0.1, lam=0.0001, k=4)
	model.init_model(ffm_data)

	for i in range(n_iter):
	    print('iteration %d, ' % i, end='')
	    model.iteration(ffm_data)

	    y_pred = model.predict(ffm_data)
	    auc = roc_auc_score(y, y_pred)
	    print('train auc %.4f' % auc)


	# save the model 
	model.save_model('ololo.bin')

	# load it to reuse the model
	model = ffm.read_model('ololo.bin')