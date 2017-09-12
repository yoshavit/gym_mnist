from setuptools import setup
import os
import time
setup(name='gym_mnist',
      version='0.0.2',
      setup_requires=['numpy','pillow>=4.2.1','six'],
      install_requires=['certifi','gym', 'opencv-python']
)
from input_data import read_data_sets
from PIL import Image
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
    im = Image.fromarray(images[i, :].reshape((28,28))).convert("RGB")
    im.show()
    time.sleep(5)
    im.close()
    exit
    im.save(data_dir + "digit" + str(l) + "/im" + str(c[l]) + ".png")
    c[l] += 1
