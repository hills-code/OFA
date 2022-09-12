# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import json
import logging
import pickle
from dataclasses import dataclass, field
from typing import Optional

from fairseq.tasks import register_task

from data.file_dataset import FileDataset
from data.unified_data.multi_unified_dataset import MultiUnifiedDataset
from data.unified_data.unified_image_cls_dataset import UnifiedImageClsDataset
from data.unified_data.unified_text_cls_dataset import UnifiedTextClsDataset
from tasks.ofa_task import OFATask, OFAConfig
from utils.trie import Trie

logger = logging.getLogger(__name__)


@dataclass
class MultiUnifiedConfig(OFAConfig):
    task_split: Optional[str] = field(
        default='text_data',
        metadata={"help": 'the split of tasks used'},
    )


@register_task("multi_task", dataclass=MultiUnifiedConfig)
class MultiUnifiedTask(OFATask):
    def __init__(self, cfg: MultiUnifiedConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        task_split_json_path = '../../utils/custom_tasks_splits/' + cfg.task_split + '.json'
        with open(task_split_json_path, 'r') as fp:
            self.tasks = json.load(fp)['dataset']

        self.data_path_prefix = '../../dataset/unified_data/'

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):

        dataset = dict()
        for data_type in self.tasks:
            if data_type == 'text':
                for task in self.tasks[data_type]:
                    path = self.data_path_prefix + data_type + '/' + task + '/' + split + '.tsv'
                    dataset[task] = UnifiedTextClsDataset(
                        split,
                        FileDataset(path, self.cfg.selected_cols),
                        self.bpe,
                        self.src_dict,
                        self.tgt_dict,
                        max_src_length=self.cfg.max_src_length,
                        max_tgt_length=self.cfg.max_tgt_length,
                        constraint_trie=self.contraint_trie[task],
                    )
            elif data_type == 'image':
                for task in self.tasks[data_type]:
                    path = self.data_path_prefix + data_type + '/' + task + '/' + split + '.tsv'
                    dataset[task] = UnifiedImageClsDataset(
                        split,
                        FileDataset(path, self.cfg.selected_cols),
                        self.bpe,
                        self.src_dict,
                        self.tgt_dict,
                        max_src_length=self.cfg.max_src_length,
                        max_tgt_length=self.cfg.max_tgt_length,
                        constraint_trie=self.contraint_trie[task],
                    )

        self.datasets[split] = MultiUnifiedDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            constraint_trie=self.contraint_trie,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        self.contraint_trie = dict()
        for data_type in self.tasks:
            for task in self.tasks[data_type]:
                constraint_trie = Trie(self.tgt_dict.eos())
                with open(self.data_path_prefix + data_type + '/' + task + '/ans2label.pkl', 'rb') as fp:
                    ans2label_dict = pickle.load(fp)
                for i, answer in enumerate(ans2label_dict.keys()):
                    answer_item = self.tgt_dict.encode_line(
                        line=self.bpe.encode(' ' + answer),
                        add_if_not_exist=False,
                        append_eos=False
                    ).long()
                    constraint_trie.insert([self.tgt_dict.bos()] + answer_item.tolist() + [self.tgt_dict.eos()])
                self.contraint_trie[task] = constraint_trie
        return model

