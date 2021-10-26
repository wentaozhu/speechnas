import os
import sys
import threading

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

if sys.version_info >= (3, 0):
    import queue as Queue
else:
    import Queue

from dataloader.utils.kaldi_io import read_mat, read_vec_flt
from config.global_enum import DataAugmentMode


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, max_prefetch=1):
        """
        This function transforms generator into a background-thead generator.
        :param generator: generator or genexp or any
        It can be used with any minibatch generator.
        It is quite lightweight, but not entirely weightless.
        Using global variables inside generator is not recommended (may rise GIL and zero-out the benefit of having a background thread.)
        The ideal use case is when everything it requires is store inside it and everything it outputs is passed through queue.
        There's no restriction on doing weird stuff, reading/writing files, retrieving URLs [or whatever] wlilst iterating.
        :param max_prefetch: defines, how many iterations (at most) can background generator keep stored at any moment of time.
        Whenever there's already max_prefetch batches stored in queue, the background process will halt until one of these batches is dequeued.
        !Default max_prefetch=1 is okay unless you deal with some weird file IO in your generator!
        Setting max_prefetch to -1 lets it store as many batches as it can, which will work slightly (if any) faster, but will require storing
        all batches in memory. If you use infinite generator with max_prefetch=-1, it will exceed the RAM size unless dequeued quickly enough.
        """
        threading.Thread.__init__(self)
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    # Python 3 compatibility
    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.generator)


class DataLoaderFast(DataLoader):
    """Use prefetch_generator to fetch batch to avoid waitting.
    """

    def __init__(self, max_prefetch, *args, **kwargs):
        assert max_prefetch >= 1
        self.max_prefetch = max_prefetch
        super(DataLoaderFast, self).__init__(*args, **kwargs)

    def __iter__(self):
        return BackgroundGenerator(super(DataLoaderFast, self).__iter__(), self.max_prefetch)


class ChunkEgs(Dataset):
    """Prepare chunk-egs for time-context-input-nnet (such as xvector which transforms egs form chunk-frames level to
    single-utterance level and TDNN) or single-channel-2D-input-nnet (such as resnet). Do it by linking to egs path
    temporarily and read them in training time, actually.
    The acoustic feature based egs are not [frames, feature-dim] matrix format any more and it should be seen as
    a [feature-dim, frames] tensor after transposing.
    """

    def __init__(self, egs_csv, egs_type="chunk", io_status=True, aug=None, aug_params={}):
        """
        @egs_csv:
            utt-id:str  ark-path:str  start-position:int  end-position:int  class-lable:int

        Other option
        @io_status: if false, do not read data from disk and return zero, which is useful for saving i/o resource
        when kipping seed index.
        """
        assert egs_type is "chunk" or egs_type is "vector"
        assert egs_csv != "" and egs_csv is not None
        head = pd.read_csv(egs_csv, sep=" ", nrows=0).columns

        assert "ark-path" in head
        assert "class-label" in head

        if egs_type is "chunk":
            if "start-position" in head and "end-position" in head:
                self.chunk_position = pd.read_csv(egs_csv, sep=" ", usecols=["start-position", "end-position"]).values
            elif "start-position" not in head and "end-position" not in head:
                self.chunk_position = None
            else:
                raise TypeError("Expected both start-position and end-position are exist in {}.".format(egs_csv))

        # It is important that using .astype(np.string_) for string object to avoid memeory leak
        # when multi-threads dataloader are used.
        self.ark_path = pd.read_csv(egs_csv, sep=" ", usecols=["ark-path"]).values.astype(np.string_)
        self.label = pd.read_csv(egs_csv, sep=" ", usecols=["class-label"]).values

        self.io_status = io_status
        self.egs_type = egs_type

        # Augmentation.
        self.aug = get_augmentation(aug, aug_params)

    def set_io_status(self, io_status):
        self.io_status = io_status

    def __getitem__(self, index):
        if not self.io_status:
            return 0., 0.

        # Decode string from bytes after using astype(np.string_).
        egs_path = str(self.ark_path[index][0], encoding='utf-8')

        if self.chunk_position is not None:
            chunk = [self.chunk_position[index][0], self.chunk_position[index][1]]
        else:
            chunk = None

        if self.egs_type is "chunk":
            egs = read_mat(egs_path, chunk=chunk)
        else:
            egs = read_vec_flt(egs_path)

        target = self.label[index][0]

        # Note, egs which is read from kaldi_io is read-only and
        # use egs = np.require(egs, requirements=['O', 'W']) to make it writeable.
        # It avoids the problem "ValueError: assignment destination is read-only".
        # Note that, do not use inputs.flags.writeable = True when the version of numpy >= 1.17.
        egs = np.require(egs, requirements=['O', 'W'])

        if self.aug is not None:
            return self.aug(egs.T), target
        else:
            return egs.T, target

    def __len__(self):
        return len(self.ark_path)


class BaseBunch(object):

    def __init__(self, dataset, dataset_type, use_fast_loader=False, max_prefetch=10,
                 batch_size=512, shuffle=True, num_workers=0, pin_memory=False, drop_last=True):
        """
        # TODO 以下是程序实际运行时，各个参数的真实值
        'use_fast_loader': True,
        'max_prefetch': 10,
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 2,
        'pin_memory': False,
        'drop_last': True
        """

        if dataset_type == "train":
            # The num_replicas/world_size and rank will be set automatically with DDP.
            train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
            num_gpu = dist.get_world_size()

            if use_fast_loader:
                # TODO 使用了train_sampler，这里的shuffle就需要设置为False !
                self.dataloader = DataLoaderFast(max_prefetch, dataset, batch_size=batch_size, shuffle=False,
                                                 num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
                                                 sampler=train_sampler)
            else:
                self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                             pin_memory=pin_memory, drop_last=drop_last, sampler=train_sampler)

            self.num_batch = len(self.dataloader)
        else:
            valid_batch_size = min(batch_size, len(dataset))  # To save GPU memory

            if len(dataset) <= 0:
                raise ValueError("Expected num_samples of valid > 0.")

            # Do not use DataLoaderFast for valid
            # for it increases the memory all the time when compute_valid_accuracy is True.
            # But I have not find the real reason.
            self.dataloader = DataLoader(dataset,
                                         batch_size=valid_batch_size,
                                         shuffle=False,
                                         num_workers=num_workers,
                                         pin_memory=pin_memory,
                                         drop_last=False)

            self.num_batch = len(self.dataloader)

    @classmethod
    def get_bunch_from_csv(self, trainset_csv: str,
                           valid_csv: str = None,
                           dataset_type: str = "",
                           egs_params: dict = {},
                           data_loader_params_dict: dict = {}):
        if dataset_type not in ["train", "val", "valid"]:
            raise Exception("get_bunch_from_csv: Unknown dataset_type !!! [Lijixiang]")

        egs_type = "chunk"
        if "egs_type" in egs_params.keys():
            egs_type = egs_params.pop("egs_type")
            if egs_type != "chunk" and egs_type != "vector":
                raise TypeError("Do not support {} egs now. Select one from [chunk, vector].".format(egs_type))

        if dataset_type == "train":
            dataset = ChunkEgs(trainset_csv, **egs_params, egs_type=egs_type)
            print("get_bunch_from_csv: get train dataset ...")
        else:
            dataset = ChunkEgs(valid_csv, egs_type=egs_type)
            print("get_bunch_from_csv: get valid dataset ...")
        return self(dataset, dataset_type, **data_loader_params_dict)

    def __len__(self):
        # main: train
        return self.num_batch

    """
    egs_params: {'aug': 'specaugment', 
                 'aug_params': {'frequency': 0.2, 
                                'frame': 0.2, 
                                'rows': 4, 
                                'cols': 4, 
                                'random_rows': True, 
                                'random_cols': True
                                }
                }
    loader_params: {'use_fast_loader': True, 
                    'max_prefetch': 10, 
                    'batch_size': 64, 
                    'shuffle': True, 
                    'num_workers': 2, 
                    'pin_memory': False, 
                    'drop_last': True}
    """

    @classmethod
    def get_bunch_from_egsdir(self, egsdir: str,
                              dataset_type: str,
                              egs_params: dict = {},
                              data_loader_params_dict: dict = {}):
        train_csv_name = None
        valid_csv_name = None

        if "train_csv_name" in egs_params.keys():
            train_csv_name = egs_params.pop("train_csv_name")

        if "valid_csv_name" in egs_params.keys():
            valid_csv_name = egs_params.pop("valid_csv_name")

        feat_dim, num_targets, train_csv, valid_csv = get_info_from_egsdir(egsdir,
                                                                           train_csv_name=train_csv_name,
                                                                           valid_csv_name=valid_csv_name)
        info = {"feat_dim": feat_dim, "num_targets": num_targets}
        bunch = self.get_bunch_from_csv(train_csv, valid_csv, dataset_type, egs_params, data_loader_params_dict)
        return bunch, info


def get_info_from_egsdir(egsdir, train_csv_name=None, valid_csv_name=None):
    if os.path.exists(egsdir + "/info"):
        feat_dim = int(read_file_to_list(egsdir + "/info/feat_dim")[0])
        num_targets = int(read_file_to_list(egsdir + "/info/num_targets")[0])

        train_csv_name = train_csv_name if train_csv_name is not None else "train.egs.csv"
        valid_csv_name = valid_csv_name if valid_csv_name is not None else "valid.egs.csv"

        # train_csv = egsdir + "/" + train_csv_name
        # valid_csv = egsdir + "/" + valid_csv_name
        train_csv = os.path.join(egsdir, train_csv_name)
        valid_csv = os.path.join(egsdir, valid_csv_name)

        # if not os.path.exists(valid_csv):
        #     valid_csv = None

        print("$$$$$$$$$$$$ The Num of VoxCeleb Samples is %d $$$$$$$$$$$" % num_targets)
        return feat_dim, num_targets, train_csv, valid_csv
    else:
        raise ValueError("Expected dir {0} to exist.".format(egsdir + "/info"))


def read_file_to_list(file_path, every_bytes=10000000):
    list = []
    with open(file_path, 'r') as reader:
        while True:
            lines = reader.readlines(every_bytes)
            if not lines:
                break
            for line in lines:
                list.append(line)
    return list


def get_augmentation(aug=None, aug_params={}):
    default_aug_params = {
        "frequency": 0.2,
        "frame": 0.2,
        "rows": 4,
        "cols": 4,
        "random_rows": True,
        "random_cols": True,
        "num_cut": 1,
        "random_cut": False
    }

    # TODO 直接写死
    aug_params = default_aug_params

    if aug is None or aug == "" or aug is False:
        return None
    elif aug == "specaugment":  # TODO aug实际值是specaugment
        print("get_augmentation: Using SpecAugment ...")
        return SpecAugment(frequency=aug_params["frequency"], frame=aug_params["frame"],
                           rows=aug_params["rows"], cols=aug_params["cols"],
                           random_rows=aug_params["random_rows"], random_cols=aug_params["random_cols"])
    elif aug == "cutout":
        raise Exception("get_augmentation: Unsupported !!! [Lijixiang]")
    else:
        raise TypeError("Do not support {} augmentation.".format(aug))


class SpecAugment(object):
    """Implement specaugment for acoustics features' augmentation but without time wraping.
    Reference: Park, D. S., Chan, W., Zhang, Y., Chiu, C.-C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019).
               Specaugment: A simple data augmentation method for automatic speech recognition. arXiv
               preprint arXiv:1904.08779.

    Likes in Compute Vision:
           [1] DeVries, T., & Taylor, G. W. (2017). Improved regularization of convolutional neural networks
               with cutout. arXiv preprint arXiv:1708.04552.

           [2] Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2017). Random erasing data augmentation.
               arXiv preprint arXiv:1708.04896.
    """

    def __init__(self, frequency=0.2, frame=0.2, rows=1, cols=1, random_rows=False, random_cols=False):
        assert 0. <= frequency < 1.
        assert 0. <= frame < 1.  # a.k.a time axis.

        self.p_f = frequency
        self.p_t = frame

        # Multi-mask.
        self.rows = rows  # Mask rows times for frequency.
        self.cols = cols  # Mask cols times for frame.

        self.random_rows = random_rows
        self.random_cols = random_cols

        self.init = False

    def __call__(self, inputs):
        """
        @inputs: a 2-dimensional tensor (a matrix), including [frenquency, time]
        """
        if self.p_f > 0. and self.p_t > 0.:
            if isinstance(inputs, np.ndarray):
                numpy_tensor = True
            elif isinstance(inputs, torch.Tensor):
                numpy_tensor = False
            else:
                raise TypeError("Expected np.ndarray or torch.Tensor, but got {}".format(type(inputs).__name__))

            if not self.init:
                input_size = inputs.shape
                assert len(input_size) == 2
                if self.p_f > 0.:
                    self.num_f = input_size[0]  # Total channels.
                    self.F = int(self.num_f * self.p_f)  # Max channels to drop.
                if self.p_t > 0.:
                    self.num_t = input_size[1]  # Total frames. It requires all egs with the same frames.
                    self.T = int(self.num_t * self.p_t)  # Max frames to drop.
                self.init = True

            if self.p_f > 0.:
                if self.random_rows:
                    multi = np.random.randint(1, self.rows + 1)
                else:
                    multi = self.rows

                for i in range(multi):
                    f = np.random.randint(0, self.F + 1)
                    f_0 = np.random.randint(0, self.num_f - f + 1)

                    inverted_factor = self.num_f / (self.num_f - f)
                    if numpy_tensor:
                        inputs[f_0:f_0 + f, :].fill(0.)
                        inputs = torch.from_numpy(inputs).mul_(inverted_factor).numpy()
                    else:
                        inputs[f_0:f_0 + f, :].fill_(0.)
                        inputs.mul_(inverted_factor)

            if self.p_t > 0.:
                if self.random_cols:
                    multi = np.random.randint(1, self.cols + 1)
                else:
                    multi = self.cols

                for i in range(multi):
                    t = np.random.randint(0, self.T + 1)
                    t_0 = np.random.randint(0, self.num_t - t + 1)

                    if numpy_tensor:
                        inputs[:, t_0:t_0 + t].fill(0.)
                    else:
                        inputs[:, t_0:t_0 + t].fill_(0.)

        return inputs


def get_vexceleb_dataset(train_batch_size,
                         test_batch_size,
                         dataset_root='./some_path/having_train_and_val_csv/',
                         dataset_type='train',
                         **kwargs):
    if dataset_type == "train":
        batchsize = train_batch_size
    elif dataset_type in ["val", "valid"]:
        batchsize = test_batch_size
    else:
        raise Exception("get_vexceleb_dataset: Unknown dataset type !!!")

    data_aug_model = kwargs["config_model"].data_augment_mode
    if data_aug_model == DataAugmentMode.No:
        print("!!!!!! VoxCeleb: Use DataAugment None !!!!!!")
        aug = None
    elif data_aug_model == DataAugmentMode.Primitive:
        print("!!!!!! VoxCeleb: Use DataAugment SpecAug !!!!!!")
        aug = "specaugment"
    else:
        raise Exception("Unknown DataAugment Mode for VoxCeleb !!!")

    egs_params = {'aug': aug,
                  'aug_params': {'frequency': 0.2,
                                 'frame': 0.2,
                                 'rows': 4,
                                 'cols': 4,
                                 'random_rows': True,
                                 'random_cols': True
                                 }
                  }
    loader_params = {'use_fast_loader': True,
                     'max_prefetch': 10,
                     'batch_size': batchsize,
                     'shuffle': True,
                     'num_workers': 2,
                     'pin_memory': False,
                     'drop_last': True
                     }
    if dataset_type == 'train':
        bunch, info = BaseBunch.get_bunch_from_egsdir(dataset_root, dataset_type, egs_params, loader_params)
        trainloader = bunch.dataloader
        print('get_vexceleb_dataset: Succeed to init VoxCeleb Train DataLoader!')
        return trainloader
    elif dataset_type == 'val' or dataset_type == 'valid':
        bunch, info = BaseBunch.get_bunch_from_egsdir(dataset_root, dataset_type, egs_params, loader_params)
        valloader = bunch.dataloader
        print('get_vexceleb_dataset: Succeed to init VoxCeleb Val DataLoader!')
        return valloader
    else:
        raise Exception('VoxCeleb DataLoader: Unknown dataset type -- %s' % dataset_type)
