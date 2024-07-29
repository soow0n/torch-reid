from __future__ import division, print_function, absolute_import
import re
import glob
import os
import os.path as osp
import warnings
import copy
from ..dataset import ImageDataset

        
        
class MARS(ImageDataset):

    _junk_pids = [0, -1]
    dataset_dir = 'MARS'
    
    def __init__(self, root, **kwargs):
        self.data_dir = osp.join(root, self.dataset_dir)
        
        self.train_dir = osp.join(self.data_dir, 'bbox_train')
        self.test_dir = osp.join(self.data_dir, 'bbox_test')
        
        required_files = [self.data_dir, self.train_dir, self.test_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        gallery = self.process_dir(self.test_dir, relabel=False)
        query = [copy.deepcopy(gallery[0])]

        super(MARS, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)C(\d)')
        
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid in self._junk_pids:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        
        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid in self._junk_pids:
                continue # junk images are just ignored
            assert 0 < pid <= 1500 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))
        
        return data

