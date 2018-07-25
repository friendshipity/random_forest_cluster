import numpy as np
import svmpy.svm as svm
import svmpy.Kernel as kernel
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import argh

num_samples=10
num_features=2
grid_size=20

samples = np.matrix(np.random.normal(size=num_samples * num_features).reshape(num_samples, num_features))
labels = 2 * (samples.sum(axis=1) > 0) - 1.0
trainer = svm.SVMTrainer(kernel.Kernel.linear(), 0.1)
predictor = trainer.train(samples, labels)