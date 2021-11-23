from glob import glob
import os
from PIL import Image


files = glob(os.path.join('./', 'test', '*/*_image.jpg'))
for i, img_file in enumerate(files):
    img = Image.open(img_file)
    img_name = os.path.split(os.path.split(img_file)[0])[1] + '_' + os.path.basename(img_file)
    img.save(os.path.join('./', 'cars', 'test', 'images', img_name))