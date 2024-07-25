from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset
import random

class Market1501(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_
    
    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = 'market1501'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(
            self, 
            root='', 
            market1501_500k=False, 
            aug_dir=None,
            aug_per_pid=0,
            aug_pid_list=[],
            split_train_into_query_gallery=False,
            train_split_ratio=0.2,
            **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        # self.dataset_dir = osp.join(root, self.dataset_dir)
        # self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        # self.data_dir = self.dataset_dir
        data_dir = osp.join(root, 'Market-1501-v15.09.15')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k
        
        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)

        # augment with generated images
        self.aug_per_pid = aug_per_pid
        self.aug_pid_list = aug_pid_list
        if self.aug_per_pid > 0:
            assert len(self.aug_pid_list) > 0, "Please select pids for augmentation"
        
        if self.aug_per_pid > 0:
            self.aug_dir = osp.join(aug_dir, f'aug{self.aug_per_pid}')
            required_files.append(self.aug_dir)
            do_aug = True
        else:
            self.aug_dir = None
            do_aug = False

        self.check_before_run(required_files)

        if split_train_into_query_gallery:
            train, pid_container = self.process_dir(self.train_dir, relabel=False, aug=False, return_pid_container=True)
            query, gallery = self.split_train_into_query_gallery(train, pid_container, train_split_ratio)
        else:
            train = self.process_dir(self.train_dir, relabel=True, aug=do_aug)
            query = self.process_dir(self.query_dir, relabel=False)
            gallery = self.process_dir(self.gallery_dir, relabel=False)
        
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, relabel=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)


    def split_train_into_query_gallery(self, train_set, pid_container, ratio=0.2):
        query, gallery = [], []
        for pid in pid_container:
            data_per_pid = [item for item in train_set if item[1] == pid]
            query_num = int(len(data_per_pid) * ratio)
            query_per_pid = random.sample(data_per_pid, query_num)
            gallery_per_pid = list(filter(lambda x: x not in query_per_pid, data_per_pid))

            query.extend(query_per_pid)
            gallery.extend(gallery_per_pid)
        
        return query, gallery


    def process_dir(self, dir_path, relabel=False, aug=False, return_pid_container=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))
        
        if aug:
            aug_data = self.process_train_aug_dir(pid_container, pid2label)
            data += aug_data
        
        if return_pid_container:
            return data, pid_container    
        
        return data
    
        
    def process_train_aug_dir(self, pid_container, pid2label):
        if self.aug_pid_list[0] == 'all':
            self.aug_pid_list = list(pid_container)
        assert set(self.aug_pid_list).issubset(pid_container)
        
        aug_data = []
        aug_paths = glob.glob(osp.join(self.aug_dir, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        for aug_path in aug_paths:
            pid, camid = map(int, pattern.search(aug_path).groups())
            if pid not in self.aug_pid_list:
                continue
            
            if pid == -1:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            pid = pid2label[pid]
            aug_data.append((aug_path, pid, camid))

        return aug_data