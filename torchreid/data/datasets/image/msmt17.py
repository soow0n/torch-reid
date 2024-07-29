from __future__ import division, print_function, absolute_import
import re
import glob
import os
import os.path as osp
import warnings

from ..dataset import ImageDataset


class MSMT17(ImageDataset):
    """MSMT17.

    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_
    
    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    dataset_dir = 'MSMT17_V2'

    def __init__(self, root, **kwargs):
        self.data_dir = osp.join(root, self.dataset_dir)

        self.train_dir = osp.join(self.data_dir, 'mask_train_v2')
        self.test_dir = osp.join(self.data_dir, 'mask_test_v2')
        
        self.list_train_path = osp.join(self.data_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.data_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.data_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.data_dir, 'list_gallery.txt')

        required_files = [self.data_dir, self.train_dir, self.test_dir]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, self.list_train_path)
        val = self.process_dir(self.train_dir, self.list_val_path)
        train += val
        
        query = self.process_dir(self.test_dir, self.list_query_path)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path)

        # Note: to fairly compare with published methods on the conventional ReID setting,
        #       do not add val images to the training set.
        # if 'combineall' in kwargs and kwargs['combineall']:
        #     train += val

        super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2]) - 1 # index starts from 0
            img_path = osp.join(dir_path, img_path)
            data.append((img_path, pid, camid))

        return data
