import layers
import numpy as np
from functions import Function
import time

class Sequential:
	def __init__(self):
		layers.sequence_ins = self
		self.sequence = []
		self.learning_rate = 0.001
		self.dtype = np.float32

	def add(self, obj):
		if len(self.sequence) > 0:
			obj(self.sequence[-1])
		self.sequence.append(obj)

	def get_inp_shape(self):
		return self.sequence[-1].shape[1:]

	def forward(self, X_inp, training=True):
		for obj in self.sequence:
			X_inp = obj.forward(X_inp, training=training)
		return X_inp

	def backprop(self, err, i):
		for obj in self.sequence[::-1]:
			err = obj.backprop(err, layer=i)
			i -= 1
		return err

	def predict(self, X_inp):
		self.svd_inp = X_inp[:1].astype(self.dtype)
		return self.forward(X_inp.astype(self.dtype), training=False)

	def train_on_batch(self, X_inp, labels):
		X_inp = self.forward(X_inp.astype(self.dtype))
		err = self.del_loss(X_inp, labels.astype(self.dtype))
		self.backprop(err, self.lenseq_m1)
		self.optimizer(self.sequence, self.learning_rate, self.beta)
		return X_inp

	def not_train_on_batch(self, X_inp, labels):
		X_inp = self.forward(X_inp.astype(self.dtype))
		err = self.del_loss(X_inp, labels.astype(self.dtype))
		err = self.backprop(err, self.lenseq_m1 + 1)
		return X_inp, err

	def fit(self, X_inp=None, labels=None, iterator=None, batch_size=1, epochs=1, validation_data=None, shuffle=True, accuracy_metric=True,
			infobeta=0.2):
		lnxinp = len(X_inp)
		acc = 0
		loss = 0
		sam_time = 0
		for epch in range(epochs):
			print("EPOCH:", epch + 1, "/", epochs)
			if iterator is None:
				s = np.random.permutation(lnxinp)
				X_inp = X_inp[s]
				labels = labels[s]
			start = time.time()
			idx = 0
			while idx < lnxinp:
				smtst = time.time()
				if iterator is not None:
					inp, y_inp = iterator.next()
				else:
					inp = X_inp[idx:idx + batch_size]
					y_inp = labels[idx:idx + batch_size]
				idx += inp.shape[0]
				logits = self.train_on_batch(inp, y_inp)
				if accuracy_metric:
					if self.loss == Function.cross_entropy_with_logits:
						ans = logits.argmax(axis=1)
						cor = y_inp.argmax(axis=1)
					else:
						ans = logits
						cor = y_inp
					nacc = (ans == cor).mean()
					acc = infobeta * nacc + (1 - infobeta) * acc
				sample_loss = self.loss(logits=logits, labels=y_inp).mean() / 10
				loss = infobeta * sample_loss + (1 - infobeta) * loss
				samtm = time.time() - smtst
				sam_time = infobeta * samtm + (1 - infobeta) * sam_time
				rem_sam = (lnxinp - idx) / batch_size
				eta = int(rem_sam * sam_time)
				print("\rProgress: {} / {}  - {}s - {:.2}s/sample - loss: {:.4f} - accuracy: {:.4f}".format(
						str(idx).rjust(6), lnxinp, eta, sam_time, sample_loss, acc), end="      _")
			end = time.time()
			print("\nEpoch time: {:.3f}s".format(end - start))
			if accuracy_metric:
				self.validate(validation_data, batch_size, infobeta)