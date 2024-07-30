from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..dataset import ImageDataset
import random
from collections import defaultdict
import os

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
            aug_dir=None,
            aug_per_pid=0,
            split_train_into_query_gallery=False,
            train_split_ratio=0.2,
            sample_mars=False,
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
        # augment with MARS dataset
        self.mars_dir = osp.join(osp.join(root, 'MARS'))
        

        required_files = [
            self.data_dir, self.train_dir, self.query_dir, self.gallery_dir, self.mars_dir
        ]
        
        # mars pid 해당 부분만 ori dataset으로 추출 => process dir
        # use mars True -> mars dataset에서 pid별로 랜덤 샘플링 후 ori dataset에 더하기 # process mars dir
        # use mars False -> mars pid 해당 폴더에서만 샘플 후 orid dataset에 더하기 # process aug dir


        # augment with generated images
        self.aug_per_pid = aug_per_pid
        if self.aug_per_pid > 0:
            self.aug_dir = osp.join(aug_dir, f'aug{self.aug_per_pid}')
            required_files.append(self.aug_dir)
        else:
            self.aug_dir = None

        self.check_before_run(required_files)
        self.mars_pid = self.get_mars_pid(self.mars_dir)
        train, pid_container, pid2label = self.process_dir(self.train_dir, relabel=True)

        mars_data = self.sample_mars_data(pid_container, pid2label)
        aug_data = self.process_aug_dir(pid_container, pid2label)
        
        if split_train_into_query_gallery:
            train, pid_container, pid2label = self.process_dir(self.train_dir, relabel=False)
            query, gallery = self.split_train_into_query_gallery(train, pid_container, train_split_ratio)
        else:
            if sample_mars:
                train += mars_data
            else:
                train += aug_data
            query, _, _ = self.process_dir(self.query_dir, relabel=False)
            gallery, _, _ = self.process_dir(self.gallery_dir, relabel=False)

        super(Market1501, self).__init__(train, query, gallery, **kwargs)


    def split_train_into_query_gallery(self, train_set, pid_container, ratio):
        query, gallery = [], []
        for pid in pid_container:
            data_per_pid = [item for item in train_set if item[1] == pid]
            query_num = int(len(data_per_pid) * ratio)
            query_per_pid = random.sample(data_per_pid, query_num)
            gallery_per_pid = list(filter(lambda x: x not in query_per_pid, data_per_pid))

            query.extend(query_per_pid)
            gallery.extend(gallery_per_pid)
        
        return query, gallery


    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1 or pid not in self.mars_pid:
                continue # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1 or pid not in pid_container:
                continue # junk images are just ignored
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))
        
        return data, pid_container, pid2label

    
    def process_aug_dir(self, pid_container, pid2label):
        if self.aug_dir is None:
            return []

        aug_data = []
        aug_paths = glob.glob(osp.join(self.aug_dir, '*.png'))
        pattern = re.compile(r'([-\d]+)_c(\d)')
        for aug_path in aug_paths:
            pid, camid = map(int, pattern.search(aug_path).groups())
            if pid == -1 or pid not in pid_container:
                continue
            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0
            pid = pid2label[pid]
            aug_data.append((aug_path, pid, camid))
        
        return aug_data
            
            
    def get_mars_pid(self, dir_path):
        pid_list = os.listdir(osp.join(dir_path, 'bbox_train')) + os.listdir(osp.join(dir_path, 'bbox_test'))
        pid_list.remove('00-1')
        pid_container = set(map(int, pid_list)) - set(self._junk_pids)
        
        return pid_container
        
    def sample_mars_data(self, pid_container, pid2label):
        pattern = re.compile(r'([-\d]+)C(\d)')
        pid_list = os.listdir(osp.join(self.mars_dir, 'bbox_train')) + os.listdir(osp.join(self.mars_dir, 'bbox_test'))
        pid_list.remove('00-1')

        data = []
        for pid in pid_list:
            if int(pid) not in pid_container:
                continue
            sampled_paths = random.sample(glob.glob(osp.join(self.mars_dir, f'*/{pid}/*.jpg')), k=self.aug_per_pid)
            for img_path in sampled_paths:
                pid, camid = map(int, pattern.search(img_path).groups())
                if pid == -1:
                    continue # junk images are just ignored
                assert 0 <= pid <= 1501 # pid == 0 means background
                assert 1 <= camid <= 6
                camid -= 1 # index starts from 0
                pid = pid2label[pid]
                data.append((img_path, pid, camid))
                
        return data
    
    def process_train_aug_dir(self, pid2label):
        self.aug_pid_list = list(self.mars_pid)
        # if self.aug_pid_list[0] == 'all':
        #     self.aug_pid_list = list(pid_container)
        # # breakpoint()
        # assert set(self.aug_pid_list).issubset(pid_container)
        
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