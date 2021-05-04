# program to create text file containing file paths of training and validation images
# this text files are used in "dota.data" 

import os
from tqdm import tqdm

data_path = "path-to-dataset"

def write_to_file(set='train'):
  file = open(f'/fred/oz138/COS80028/P2/rakesh/yoltv4/{set}.txt', 'a+')
  for i in tqdm(os.listdir(datapath + set + '/images')):
    file.write(datapath + set + '/images' + f'{i}\n')
    
  file.close()
    
if __name__ == "__main__":
  write_to_file('train')
  write_to_file('val')
