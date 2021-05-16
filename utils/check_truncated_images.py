'''
Program to check truncated images after image split
'''
import os
from tqdm import tqdm
from PIL import Image

print('checking train images')
c = 0
for filename in tqdm(os.listdir('/fred/oz138/COS80028/P2/rakesh/data/train/train_split/images')):
  try:
		im = Image.open(f'/fred/oz138/COS80028/P2/rakesh/data/train/train_split/images/{filename}')
		im.verify()
	except:
		print(filename)
		c += 1
    
print(c)

cc =0
print('checking valid images')
for filename in tqdm(os.listdir('/fred/oz138/COS80028/P2/rakesh/data/val/val_split/images')):
	try:
		im = Image.open(f'/fred/oz138/COS80028/P2/rakesh/data/val/val_split/images/{filename}')
		im.verify()
	except:
		print(filename)
		cc+=1

print(cc)
