"""
Script to reduce the number of classes to train for the DOTA dataset
"""
import os
import shutil
from tqdm import tqdm

objects = ['small-vehicle',
		   'large-vehicle',
		   'plane',
		   'ship',
		   'bridge',
		   'harbor']

def preprocess(set='train'):
	os.mkdir(f'/fred/oz138/COS80028/P2/rakesh/data/{set}/newlabelTxt/')

	for txt in tqdm(os.listdir(f'/fred/oz138/COS80028/P2/rakesh/data/{set}/labelTxt')):
		file = open(f'/fred/oz138/COS80028/P2/rakesh/data/{set}/labelTxt/'+txt, 'r').readlines()
		new_file = open(f'/fred/oz138/COS80028/P2/rakesh/data/{set}/newlabelTxt/'+txt, 'a+')
		c = 0
		for line in file:
			if c >= 2:
				i = line.split()[-2]
				if i in objects:
					new_file.write(f'{line}')
				else:
					pass
			else:
				c += 1
				pass
		new_file.close()

	for txt in tqdm(os.listdir(f'/fred/oz138/COS80028/P2/rakesh/data/{set}/newlabelTxt')):
		file = open(f'/fred/oz138/COS80028/P2/rakesh/data/{set}/newlabelTxt/'+txt, 'r').readlines()
		if len(file) == 0:
			os.remove(f'/fred/oz138/COS80028/P2/rakesh/data/{set}/newlabelTxt/'+txt)

	os.mkdir(f'/fred/oz138/COS80028/P2/rakesh/data/{set}/newImages/')

	img_list = [i.split('.txt')[0] for i in os.listdir(f'/fred/oz138/COS80028/P2/rakesh/data/{set}/newlabelTxt/')]
	for img in tqdm(os.listdir(f'/fred/oz138/COS80028/P2/rakesh/data/{set}/images')):
		file_name = img.split('.png')[0]
		if file_name in img_list:
			shutil.copy(f'/fred/oz138/COS80028/P2/rakesh/data/{set}/images/{file_name}.png', f'/fred/oz138/COS80028/P2/rakesh/data/{set}/newImages/{file_name}.png')


if __name__ == "__main__":
	preprocess('train')
	preprocess('val')
