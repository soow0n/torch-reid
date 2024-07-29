from __future__ import division, print_function, absolute_import
import re
import glob
import os
import os.path as osp
import warnings
import copy
from ..dataset import ImageDataset


class CUHK03(ImageDataset):


    dataset_dir = 'cuhk03'

    def __init__(self, root, **kwargs):
        self.data_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.data_dir, 'images')
    
        required_files = [self.data_dir, self.train_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        gallery = [copy.deepcopy(train[0])]
        query = [copy.deepcopy(train[0])]

        super(CUHK03, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_([\d]+)_([\d]+)')
        
        data = []
        for img_path in img_paths:
            pid, camid, _ = map(int, pattern.search(img_path).groups())
            data.append((img_path, pid, camid))
            
        return data