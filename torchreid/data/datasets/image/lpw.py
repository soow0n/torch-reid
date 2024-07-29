from __future__ import division, print_function, absolute_import
import re
import glob
import os
import os.path as osp
import warnings

from ..dataset import ImageDataset

class LPW(ImageDataset):
    
    dataset_dir = 'LPW'
    
    def __init__(self, root, **kwargs):
        self.data_dir = osp.join(root, self.dataset_dir)
        required_files = [self.data_dir]
        self.check_before_run(required_files)
        
        train, query, gallery = self.process_dir(self.data_dir)
        super(LPW, self).__init__(train, query, gallery, **kwargs)
        
    def process_dir(self, dir_path):
        pid_container = set()
        camid_container = set()
        for img_dir in glob.glob(osp.join(dir_path, '/*/*/*')):
            scene, view, pid = img_dir.split('/')[-3:]
            pid_container.add((scene, pid))
            camid_container.add((scene, view))
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        camid2label = {camid: label for label, camid in enumerate(camid_container)}
       
        train_dir = ['scen2', 'scen3']
        query_dir = ['scen1/view2']
        gallery_dir = ['scen1/view1', 'scen1/view3']
        
        train, query, gallery = [], [], []
        for img_path in glob.glob(osp.join(dir_path, '/*/*/*/*.jpg')):
            scene, view, pid, frame = img_path.split('/')[-4:]            
            pid = pid2label[(scene, pid)]
            camid = camid2label[(scene, view)]
            
            item = (img_path, pid, camid)
            if scene in train_dir:
                train.append(item)
            elif f'{scene}/{view}' in query_dir:
                query.append(item)
            elif f'{scene}/{view}' in gallery_dir:
                gallery.append(item)
                
        return train, query, gallery
        
