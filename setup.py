from setuptools import setup
import os
from scipy.misc import imsave
from input_data import read_data_sets

setup(name='gym_mnist',
      version='0.0.2',
      install_requires=['certifi','gym','pillow']
)
this_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = this_dir + "/gym_mnist/resources/"
zip_dir = data_dir + "zipfiles/"
if not os.path.exists(zip_dir):
    os.makedirs(zip_dir)
digit_dirs = [data_dir + "digit" + str(i) + "/" for i in range(10)]
for dir in digit_dirs:
    if not os.path.exists(dir):
        os.makedirs(dir)
print ("Downloading MNIST dataset...")
data_sets = read_data_sets(zip_dir)
images = data_sets.train.images
labels = data_sets.train.labels
c = [0]*10
print ("Saving assorted MNIST digits to disk...")
for i in range(len(images)):
    l = labels[i]
    im = images[i, :].reshape((28,28))
    imsave(data_dir + "digit" + str(l) + "/im" + str(c[l]) + ".png", im)
    c[l] += 1

