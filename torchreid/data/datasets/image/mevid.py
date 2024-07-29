from __future__ import division, print_function, absolute_import
import re
import glob
import os
import os.path as osp
import warnings
import copy
from ..dataset import ImageDataset

class MEVID(ImageDataset):
    
    dataset_dir = 'MEVID'
    
    def __init__(self, root, **kwargs):
        self.data_dir = osp.join(root, self.dataset_dir)
        
        self.train_dir = osp.join(self.data_dir, 'bbox_train')
        self.test_dir = osp.join(self.data_dir, 'bbox_test')
        
        required_files = [self.data_dir, self.train_dir, self.test_dir]
        self.check_before_run(required_files)
        
        train = self.process_dir(self.train_dir)
        gallery = self.process_dir(self.test_dir)
        query = [copy.deepcopy(gallery[0])]
        
        super(MEVID, self).__init__(train, query, gallery, **kwargs)
        
    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))
        pattern = re.compile(r'([-\d]+)O([\d]+)C([\d]+)')
        
        pid_container = set()
        for img_path in img_paths:
            pid, outfitid, camid = map(int, pattern.search(img_path).groups())    
            pid_container.add((pid, outfitid))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, outfitid, camid = map(int, pattern.search(img_path).groups())  
            pid = pid2label[(pid, outfitid)]
            data.append((img_path, pid, camid))
        
        return data
        
