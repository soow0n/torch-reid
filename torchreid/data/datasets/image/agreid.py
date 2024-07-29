from __future__ import division, print_function, absolute_import
import re
import glob
import os
import os.path as osp
import warnings
import copy
from ..dataset import ImageDataset

class AGReIDV2(ImageDataset):
    
    dataset_dir = 'AG-ReID.v2'
    
    def __init__(self, root, **kwargs):
        self.data_dir = osp.join(root, self.dataset_dir)
        
        self.train_dir = osp.join(self.data_dir, 'train_all')
        self.gallery_dir = osp.join(self.data_dir, 'gallery')
        
        required_files = [self.data_dir, self.train_dir, self.gallery_dir]
        self.check_before_run(required_files)
        
        train = self.process_dir(self.train_dir)
        gallery = self.process_dir(self.gallery_dir)
        query = [copy.deepcopy(gallery[0])]
        
        super(AGReIDV2, self).__init__(train, query, gallery, **kwargs)
        
    def process_dir(self, dir_path):
        img_paths = glob.glob(osp.join(dir_path, '*/*.jpg'))
        pattern = re.compile(r'P([-\d]+)T([\d]+)A(\d)C(\d)')

        pid_container = os.listdir(dir_path)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        data = []
        for img_path in img_paths:
            _, _, _, camid = map(int, pattern.search(img_path).groups())
            pid = pid2label[img_path.split('/')[-2]]
            data.append((img_path, pid, camid))
        
        return data
        
