from setuptools import setup

setup(name='gym_mnist',
      version='0.0.1',
      install_requires=['gym'] 
)
# try:
    # from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
    # data_sets = read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
# except:
    # raise NotImplementedError('Currently requires Tensorflow as dependency to
                              # import MNIST.')
