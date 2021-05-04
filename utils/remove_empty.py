import os
import shutil
from tqdm import tqdm

# remove all empty text annotation files after image split
path = "path-to-cropped-labelText"
for i in os.listdir(path):
  if os.stat(path + i).st_size == 0:
    os.remove(path + i)
  else:
    pass
  
# remove all corresponding images with empty text annotation file
text_list = os.listdir(path)
text_list_strip = [i.split('.txt')[0] for i in text_list]

img_path = "path-to-images"
for i in tqdm(os.listdir(img_path)):
  if i.split('.png')[0] not in text_list_strip:
    os.remove(img_path + i)
  else:
    pass
