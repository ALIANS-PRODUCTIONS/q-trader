import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
			tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)

policy=mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

#from tf.keras.models import Sequential
#from tf.keras.models import load_model
#from tf.keras.layers import Dense
#from tf.keras.optimizers import Adam

import numpy as np
import random
from collections import deque

class Agent:
	def __init__(self, state_size, is_eval=False, model_name=""):
		self.state_size = state_size # normalized previous days
		self.action_size = 3 # sit, buy, sell
		self.memory = deque(maxlen=1000)
		self.inventory = []
		self.model_name = model_name
		self.is_eval = is_eval

		self.gamma = 0.95
		self.epsilon = 1.0
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995

		self.model = tf.keras.models.load_model("models/" + model_name) if is_eval else self._model()

	def _model(self):
		#model = Sequential()
		#model.add(Dense(units=64, input_dim=self.state_size, activation="relu"))
		#model.add(Dense(units=32, activation="relu"))
		#model.add(Dense(units=8, activation="relu"))
		#model.add(Dense(self.action_size, activation="linear"))

		model = tf.keras.models.Sequential([
			tf.keras.layers.Dense(64,activation='relu', input_shape=(self.state_size,)),
			tf.keras.layers.Dense(32, activation='relu'),
			tf.keras.layers.Dense(8, activation='relu'),
			tf.keras.layers.Dense(self.action_size, activation='relu')])
		#model.compile(loss="mse", optimizer=Adam(lr=0.001))
		model.compile(optimizer=Adam(lr=0.001), loss='mse')
		return model

	def act(self, state):
		if not self.is_eval and np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)

		options = self.model.predict(state)
		return np.argmax(options[0])

	def expReplay(self, batch_size):
		mini_batch = []
		l = len(self.memory)
		for i in range(l - batch_size + 1, l):
			mini_batch.append(self.memory[i])

		for state, action, reward, next_state, done in mini_batch:
			target = reward
			if not done:
				target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

			target_f = self.model.predict(state)
			target_f[0][action] = target
			self.model.fit(state, target_f, epochs=1, verbose=0)

		if self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay 
