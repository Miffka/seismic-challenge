import os.path as osp
import os
import torch

class Config():
    basedir = osp.abspath(osp.dirname(__file__))
    CURRENT_PATH = osp.dirname(osp.realpath(__file__))
    root_dir = osp.join(CURRENT_PATH, '..')

    models_dir = osp.join(root_dir, 'models')
    data_dir = osp.join(root_dir, 'data')
    submissions_dir = osp.join(root_dir, 'submissions')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()
