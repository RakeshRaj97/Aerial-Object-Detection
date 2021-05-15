import cvtools

'''
Set the path to point the train and valid directories
├── train
│   ├── images
│   ├── labelTxt
└── val
    ├── images
    ├── labelTxt
'''

label_root = 'path-to-labeTxts-train/val'
image_root = 'path-to-images-train/val'

dota_to_coco = cvtools.DOTA2COCO(label_root, image_root)
dota_to_coco.convert()

'''
Final directory structure looks like...
.
├── train
│   ├── images
│   ├── labelTxt
│   └── train.json
└── val
    ├── images
    ├── labelTxt
    └── valid.json
'''

save = 'file-name-to-save-train/valid-json-format'
dota_to_coco.save_json(save)

