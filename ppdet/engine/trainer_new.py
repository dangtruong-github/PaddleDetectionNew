# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import torchvision

import os
import sys
import copy
import time
import yaml
from tqdm import tqdm

import json

import numpy as np
import typing
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import paddle.nn.functional as F

import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.static import InputSpec
from ppdet.optimizer import ModelEMA

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight, convert_to_dict
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.metrics import get_infer_results, KeyPointTopDownCOCOEval, KeyPointTopDownCOCOWholeBadyHandEval, KeyPointTopDownMPIIEval, Pose3DEval
from ppdet.metrics import Metric, COCOMetric, VOCMetric, WiderFaceMetric, RBoxMetric, JDEDetMetric, SNIPERCOCOMetric, CULaneMetric
from ppdet.data.source.sniper_coco import SniperCOCODataSet
from ppdet.data.source.category import get_categories
import ppdet.utils.stats as stats
from ppdet.utils.fuse_utils import fuse_conv_bn
from ppdet.utils import profiler
from ppdet.modeling.post_process import multiclass_nms
from ppdet.modeling.lane_utils import imshow_lanes

from .callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer, WiferFaceEval, VisualDLWriter, SniperProposalsGenerator, WandbCallback, SemiCheckpointer, SemiLogPrinter
from .export_utils import _dump_infer_config, _prune_input_spec, apply_to_static
from .naive_sync_bn import convert_syncbn

from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet.engine')

__all__ = ['Trainer']

MOT_ARCH = ['JDE', 'FairMOT', 'DeepSORT', 'ByteTrack', 'CenterTrack']


class Trainer(object):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg.copy()
        assert mode.lower() in ['train', 'eval', 'test'], \
                "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()
        self.optimizer = None
        self.is_loaded_weights = False
        self.use_amp = self.cfg.get('amp', False)
        self.amp_level = self.cfg.get('amp_level', 'O1')
        self.custom_white_list = self.cfg.get('custom_white_list', None)
        self.custom_black_list = self.cfg.get('custom_black_list', None)
        self.use_master_grad = self.cfg.get('master_grad', False)
        self.uniform_output_enabled = self.cfg.get('uniform_output_enabled', False)
        if ('slim' in cfg and cfg['slim_type'] == 'PTQ') or self.uniform_output_enabled:
            self.cfg['TestDataset'] = create('TestDataset')()
        log_ranks = cfg.get('log_ranks', '0')
        if isinstance(log_ranks, str):
            self.log_ranks = [int(i) for i in log_ranks.split(',')]
        elif isinstance(log_ranks, int):
            self.log_ranks = [log_ranks]
        train_results_path = os.path.abspath(os.path.join(self.cfg.save_dir, "train_results.json"))
        if self.uniform_output_enabled:
            if os.path.exists(train_results_path) and self.mode == 'train':
                try:
                    os.remove(train_results_path)
                except:
                    pass
            if not os.path.exists(self.cfg.save_dir):
                os.mkdir(self.cfg.save_dir)
            with open(os.path.join(self.cfg.save_dir, "config.yaml"), "w") as f:
                config_dict = convert_to_dict(self.cfg)
                config_dict = {k: v for k, v in config_dict.items() if v != {}}
                yaml.dump(config_dict, f)

        # build data loader
        capital_mode = self.mode.capitalize()
        if cfg.architecture in MOT_ARCH and self.mode in [
                'eval', 'test'
        ] and cfg.metric not in ['COCO', 'VOC']:
            self.dataset = self.cfg['{}MOTDataset'.format(
                capital_mode)] = create('{}MOTDataset'.format(capital_mode))()
        else:
            self.dataset = self.cfg['{}Dataset'.format(capital_mode)] = create(
                '{}Dataset'.format(capital_mode))()

        if cfg.architecture == 'DeepSORT' and self.mode == 'train':
            logger.error('DeepSORT has no need of training on mot dataset.')
            sys.exit(1)

        if cfg.architecture == 'FairMOT' and self.mode == 'eval':
            images = self.parse_mot_images(cfg)
            self.dataset.set_images(images)

        if self.mode == 'train':
            # print(f"capital_modes in trainer: {capital_mode}")
            # print(f"self.dataset in trainer: {self.dataset}")
            self.loader = create('{}Reader'.format(capital_mode))(
                self.dataset, cfg.worker_num)
            #for step_id, data in enumerate(self.loader):
                #print(f"data in trainer.py: {data.keys()}")
                #raise Exception("DEBUG")
            # print(f"item of self.loader: {self.loader[0].keys()}")

        if cfg.architecture == 'JDE' and self.mode == 'train':
            self.cfg['JDEEmbeddingHead'][
                'num_identities'] = self.dataset.num_identities_dict[0]
            # JDE only support single class MOT now.

        if cfg.architecture == 'FairMOT' and self.mode == 'train':
            self.cfg['FairMOTEmbeddingHead'][
                'num_identities_dict'] = self.dataset.num_identities_dict
            # FairMOT support single class and multi-class MOT now.

        # build model
        if 'model' not in self.cfg:
            self.model = create(cfg.architecture)
        else:
            self.model = self.cfg.model
            self.is_loaded_weights = True

        if cfg.architecture == 'YOLOX':
            for k, m in self.model.named_sublayers():
                if isinstance(m, nn.BatchNorm2D):
                    m._epsilon = 1e-3  # for amp(fp16)
                    m._momentum = 0.97  # 0.03 in pytorch

        # reset norm param attr for setting them in optimizer
        if 'reset_norm_param_attr' in cfg and cfg['reset_norm_param_attr']:
            self.model = self.reset_norm_param_attr(
                self.model, weight_attr=None, bias_attr=None)

        # normalize params for deploy
        if 'slim' in cfg and cfg['slim_type'] == 'OFA':
            self.model.model.load_meanstd(cfg['TestReader'][
                'sample_transforms'])
        elif 'slim' in cfg and cfg['slim_type'] == 'Distill':
            self.model.student_model.load_meanstd(cfg['TestReader'][
                'sample_transforms'])
        elif 'slim' in cfg and cfg[
                'slim_type'] == 'DistillPrune' and self.mode == 'train':
            self.model.student_model.load_meanstd(cfg['TestReader'][
                'sample_transforms'])
        else:
            self.model.load_meanstd(cfg['TestReader']['sample_transforms'])

        # EvalDataset build with BatchSampler to evaluate in single device
        # TODO: multi-device evaluate
        if self.mode == 'eval':
            if cfg.architecture == 'FairMOT':
                self.loader = create('EvalMOTReader')(self.dataset, 0)
            elif cfg.architecture == "METRO_Body":
                reader_name = '{}Reader'.format(self.mode.capitalize())
                self.loader = create(reader_name)(self.dataset, cfg.worker_num)
            else:
                self._eval_batch_sampler = paddle.io.BatchSampler(
                    self.dataset, batch_size=self.cfg.EvalReader['batch_size'])
                reader_name = '{}Reader'.format(self.mode.capitalize())
                # If metric is VOC, need to be set collate_batch=False.
                if cfg.metric == 'VOC':
                    self.cfg[reader_name]['collate_batch'] = False
                self.loader = create(reader_name)(self.dataset, cfg.worker_num,
                                                  self._eval_batch_sampler)
        # TestDataset build after user set images, skip loader creation here

        # get Params
        print_params = self.cfg.get('print_params', False)
        if print_params:
            params = sum([
                p.numel() for n, p in self.model.named_parameters()
                if all([x not in n for x in ['_mean', '_variance', 'aux_']])
            ])  # exclude BatchNorm running status
            logger.info('Model Params : {} M.'.format((params / 1e6).numpy()[
                0]))

        # build optimizer in train mode
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            if steps_per_epoch < 1:
                logger.warning(
                    "Samples in dataset are less than batch_size, please set smaller batch_size in TrainReader."
                )
            self.lr = create('LearningRate')(steps_per_epoch)
            self.optimizer = create('OptimizerBuilder')(self.lr, self.model)

            # Unstructured pruner is only enabled in the train mode.
            if self.cfg.get('unstructured_prune'):
                self.pruner = create('UnstructuredPruner')(self.model,
                                                           steps_per_epoch)
        if self.use_amp and self.amp_level == 'O2':
            paddle_version = paddle.__version__[:3]
            # paddle version >= 2.5.0 or develop
            if paddle_version in ["2.5", "0.0"]:
                self.model, self.optimizer = paddle.amp.decorate(
                    models=self.model,
                    optimizers=self.optimizer,
                    level=self.amp_level,
                    master_grad=self.use_master_grad)
            else:
                self.model, self.optimizer = paddle.amp.decorate(
                    models=self.model,
                    optimizers=self.optimizer,
                    level=self.amp_level)

        # support sync_bn for npu/xpu
        if (paddle.get_device()[:3]=='npu' or paddle.get_device()[:3]=='xpu'):
            use_npu = ('use_npu' in cfg and cfg['use_npu'])
            use_xpu = ('use_xpu' in cfg and cfg['use_xpu'])
            use_mlu = ('use_mlu' in cfg and cfg['use_mlu'])
            norm_type = ('norm_type' in cfg and cfg['norm_type'])
            if norm_type == 'sync_bn' and (use_npu or use_xpu or use_mlu) and dist.get_world_size() > 1:
                convert_syncbn(self.model)

        self.use_ema = ('use_ema' in cfg and cfg['use_ema'])
        if self.use_ema:
            ema_decay = self.cfg.get('ema_decay', 0.9998)
            ema_decay_type = self.cfg.get('ema_decay_type', 'threshold')
            cycle_epoch = self.cfg.get('cycle_epoch', -1)
            ema_black_list = self.cfg.get('ema_black_list', None)
            ema_filter_no_grad = self.cfg.get('ema_filter_no_grad', False)
            self.ema = ModelEMA(
                self.model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch,
                ema_black_list=ema_black_list,
                ema_filter_no_grad=ema_filter_no_grad)

        self._nranks = dist.get_world_size()
        self._local_rank = dist.get_rank()

        self.status = {}

        self.start_epoch = 0
        self.end_epoch = 0 if 'epoch' not in cfg else cfg.epoch

        # initial default callbacks
        self._init_callbacks()

        # initial default metrics
        self._init_metrics()
        self._reset_metrics()

    def _init_callbacks(self):
        if self.mode == 'train':
            if self.cfg.get('ssod_method',
                            False) and self.cfg['ssod_method'] == 'Semi_RTDETR':
                self._callbacks = [SemiLogPrinter(self), SemiCheckpointer(self)]
            else:
                self._callbacks = [LogPrinter(self), Checkpointer(self)]
            if self.cfg.get('use_vdl', False):
                self._callbacks.append(VisualDLWriter(self))
            if self.cfg.get('save_proposals', False):
                self._callbacks.append(SniperProposalsGenerator(self))
            if self.cfg.get('use_wandb', False) or 'wandb' in self.cfg:
                self._callbacks.append(WandbCallback(self))
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'eval':
            self._callbacks = [LogPrinter(self)]
            # if self.cfg.metric == 'WiderFace':
            #     self._callbacks.append(WiferFaceEval(self))
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'test' and self.cfg.get('use_vdl', False):
            self._callbacks = [VisualDLWriter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        else:
            self._callbacks = []
            self._compose_callback = None

    def _init_metrics(self, validate=False):
        if self.mode == 'test' or (self.mode == 'train' and not validate):
            self._metrics = []
            return
        classwise = self.cfg['classwise'] if 'classwise' in self.cfg else False
        if self.cfg.metric == 'COCO' or self.cfg.metric == "SNIPERCOCO":
            # TODO: bias should be unified
            bias = 1 if self.cfg.get('bias', False) else 0
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)

            # pass clsid2catid info to metric instance to avoid multiple loading
            # annotation file
            clsid2catid = {v: k for k, v in self.dataset.catid2clsid.items()} \
                                if self.mode == 'eval' else None

            save_threshold = self.cfg.get('save_threshold', 0)

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            if self.mode == 'train' and validate:
                eval_dataset = self.cfg['EvalDataset']
                eval_dataset.check_or_download_dataset()
                anno_file = eval_dataset.get_anno()
                dataset = eval_dataset
            else:
                dataset = self.dataset
                anno_file = dataset.get_anno()

            IouType = self.cfg['IouType'] if 'IouType' in self.cfg else 'bbox'
            if self.cfg.metric == "COCO":
                self._metrics = [
                    COCOMetric(
                        anno_file=anno_file,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only,
                        save_threshold=save_threshold)
                ]
            elif self.cfg.metric == "SNIPERCOCO":  # sniper
                self._metrics = [
                    SNIPERCOCOMetric(
                        anno_file=anno_file,
                        dataset=dataset,
                        clsid2catid=clsid2catid,
                        classwise=classwise,
                        output_eval=output_eval,
                        bias=bias,
                        IouType=IouType,
                        save_prediction_only=save_prediction_only)
                ]
        elif self.cfg.metric == 'RBOX':
            # TODO: bias should be unified
            bias = self.cfg['bias'] if 'bias' in self.cfg else 0
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            imid2path = self.cfg.get('imid2path', None)

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            anno_file = self.dataset.get_anno()
            if self.mode == 'train' and validate:
                eval_dataset = self.cfg['EvalDataset']
                eval_dataset.check_or_download_dataset()
                anno_file = eval_dataset.get_anno()

            self._metrics = [
                RBoxMetric(
                    anno_file=anno_file,
                    classwise=classwise,
                    output_eval=output_eval,
                    bias=bias,
                    save_prediction_only=save_prediction_only,
                    imid2path=imid2path)
            ]
        elif self.cfg.metric == 'VOC':
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)

            self._metrics = [
                VOCMetric(
                    label_list=self.dataset.get_label_list(),
                    class_num=self.cfg.num_classes,
                    map_type=self.cfg.map_type,
                    classwise=classwise,
                    output_eval=output_eval,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg.metric == 'WiderFace':
            self._metrics = [
                WiderFaceMetric()
            ]
        elif self.cfg.metric == 'KeyPointTopDownCOCOEval':
            eval_dataset = self.cfg['EvalDataset']
            eval_dataset.check_or_download_dataset()
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownCOCOEval(
                    anno_file,
                    len(eval_dataset),
                    self.cfg.num_joints,
                    self.cfg.save_dir,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg.metric == 'KeyPointTopDownCOCOWholeBadyHandEval':
            eval_dataset = self.cfg['EvalDataset']
            eval_dataset.check_or_download_dataset()
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownCOCOWholeBadyHandEval(
                    anno_file,
                    len(eval_dataset),
                    self.cfg.num_joints,
                    self.cfg.save_dir,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg.metric == 'KeyPointTopDownMPIIEval':
            eval_dataset = self.cfg['EvalDataset']
            eval_dataset.check_or_download_dataset()
            anno_file = eval_dataset.get_anno()
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            self._metrics = [
                KeyPointTopDownMPIIEval(
                    anno_file,
                    len(eval_dataset),
                    self.cfg.num_joints,
                    self.cfg.save_dir,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg.metric == 'Pose3DEval':
            save_prediction_only = self.cfg.get('save_prediction_only', False)
            self._metrics = [
                Pose3DEval(
                    self.cfg.save_dir,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg.metric == 'MOTDet':
            self._metrics = [JDEDetMetric(), ]
        elif self.cfg.metric == 'CULaneMetric':
            output_eval = self.cfg.get('output_eval', None)
            self._metrics = [
                CULaneMetric(
                    cfg=self.cfg,
                    output_eval=output_eval,
                    split=self.dataset.split,
                    dataset_dir=self.cfg.dataset_dir)
            ]
        else:
            logger.warning("Metric not support for metric type {}".format(
                self.cfg.metric))
            self._metrics = []

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def register_callbacks(self, callbacks):
        callbacks = [c for c in list(callbacks) if c is not None]
        for c in callbacks:
            assert isinstance(c, Callback), \
                    "metrics shoule be instances of subclass of Metric"
        self._callbacks.extend(callbacks)
        self._compose_callback = ComposeCallback(self._callbacks)

    def register_metrics(self, metrics):
        metrics = [m for m in list(metrics) if m is not None]
        for m in metrics:
            assert isinstance(m, Metric), \
                    "metrics shoule be instances of subclass of Metric"
        self._metrics.extend(metrics)

    def load_weights(self, weights, ARSL_eval=False):
        if self.is_loaded_weights:
            return
        self.start_epoch = 0
        load_pretrain_weight(self.model, weights, ARSL_eval)
        logger.debug("Load weights {} to start training".format(weights))

    def load_weights_sde(self, det_weights, reid_weights):
        if self.model.detector:
            load_weight(self.model.detector, det_weights)
            if self.model.reid:
                load_weight(self.model.reid, reid_weights)
        else:
            load_weight(self.model.reid, reid_weights)

    def resume_weights(self, weights):
        # support Distill resume weights
        if hasattr(self.model, 'student_model'):
            self.start_epoch = load_weight(self.model.student_model, weights,
                                           self.optimizer)
        else:
            self.start_epoch = load_weight(self.model, weights, self.optimizer,
                                           self.ema if self.use_ema else None)
        logger.debug("Resume weights of epoch {}".format(self.start_epoch))

    def train(self, validate=False):
        assert self.mode == 'train', "Model not in 'train' mode"
        Init_mark = False
        if validate:
            self.cfg['EvalDataset'] = self.cfg.EvalDataset = create(
                "EvalDataset")()

        model = self.model
        #print(f"model: {model}")
        if self.cfg.get('to_static', False):
            model = apply_to_static(self.cfg, model)
        sync_bn = (getattr(self.cfg, 'norm_type', None) == 'sync_bn' and
                   self.cfg.use_gpu and self._nranks > 1)
        if sync_bn:
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # enabel auto mixed precision mode
        if self.use_amp:
            scaler = paddle.amp.GradScaler(
                enable=self.cfg.use_gpu or self.cfg.use_npu or self.cfg.use_mlu,
                init_loss_scaling=self.cfg.get('init_loss_scaling', 1024))
        # get distributed model
        if self.cfg.get('fleet', False):
            model = fleet.distributed_model(model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg[
                'find_unused_parameters'] if 'find_unused_parameters' in self.cfg else False
            model = paddle.DataParallel(
                model, find_unused_parameters=find_unused_parameters)

        self.status.update({
            'epoch_id': self.start_epoch,
            'step_id': 0,
            'steps_per_epoch': len(self.loader)
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num)
            self._flops(flops_loader)
        profiler_options = self.cfg.get('profiler_options', None)

        self._compose_callback.on_train_begin(self.status)

        use_fused_allreduce_gradients = self.cfg[
            'use_fused_allreduce_gradients'] if 'use_fused_allreduce_gradients' in self.cfg else False

        for epoch_id in range(self.start_epoch, self.cfg.epoch):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self._compose_callback.on_epoch_begin(self.status)
            self.loader.dataset.set_epoch(epoch_id)
            model.train()
            iter_tic = time.time()
            for step_id, data in enumerate(self.loader):
                #print(f"data in trainer.py: {data.keys()}")
                #print(f"data[image] in trainer.py: {data['image'].shape}")
                #print(f"data[im_shape] in trainer.py: {data['im_shape']}")
                #print(f"data[gt_bbox] in trainer.py: {data['gt_bbox']}")
                def deep_pin(blob, blocking):
                    if isinstance(blob, paddle.Tensor):
                        return blob.cuda(blocking=blocking)
                    elif isinstance(blob, dict):
                        return {k: deep_pin(v, blocking) for k, v in blob.items()}
                    elif isinstance(blob, (list, tuple)):
                        return type(blob)([deep_pin(x, blocking) for x in blob])
                    else:
                        return blob
                if paddle.base.core.is_compiled_with_cuda():
                    data = deep_pin(data, False)

                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id
                profiler.add_profiler_step(profiler_options)
                self._compose_callback.on_step_begin(self.status)
                data['epoch_id'] = epoch_id
                if self.cfg.get('to_static',
                                False) and 'image_file' in data.keys():
                    data.pop('image_file')

                if self.use_amp:
                    if isinstance(
                            model, paddle.
                            DataParallel) and use_fused_allreduce_gradients:
                        with model.no_sync():
                            with paddle.amp.auto_cast(
                                    enable=self.cfg.use_gpu or
                                    self.cfg.use_npu or self.cfg.use_mlu,
                                    custom_white_list=self.custom_white_list,
                                    custom_black_list=self.custom_black_list,
                                    level=self.amp_level):
                                # model forward
                                outputs = model(data)
                                loss = outputs['loss']
                            # model backward
                            scaled_loss = scaler.scale(loss)
                            scaled_loss.backward()
                        fused_allreduce_gradients(
                            list(model.parameters()), None)
                    else:
                        with paddle.amp.auto_cast(
                                enable=self.cfg.use_gpu or self.cfg.use_npu or
                                self.cfg.use_mlu,
                                custom_white_list=self.custom_white_list,
                                custom_black_list=self.custom_black_list,
                                level=self.amp_level):
                            # model forward
                            outputs = model(data)
                            #print(f"data: {data}")
                            #print(f"outputs: {outputs}")
                            #for key in outputs:
                            #    print(f"{key}: {outputs[key].detach().cpu().#numpy()}")
                            #raise Exception("DEBUG")
                            loss = outputs['loss']
                        # model backward
                        scaled_loss = scaler.scale(loss)
                        scaled_loss.backward()
                    # in dygraph mode, optimizer.minimize is equal to optimizer.step
                    scaler.minimize(self.optimizer, scaled_loss)
                else:
                    if isinstance(
                            model, paddle.
                            DataParallel) and use_fused_allreduce_gradients:
                        with model.no_sync():
                            # model forward
                            outputs = model(data)
                            loss = outputs['loss']
                            # model backward
                            loss.backward()
                        fused_allreduce_gradients(
                            list(model.parameters()), None)
                    else:
                        # model forward
                        outputs = model(data)
                        loss = outputs['loss']
                        # model backward
                        loss.backward()
                    self.optimizer.step()
                curr_lr = self.optimizer.get_lr()
                self.lr.step()
                if self.cfg.get('unstructured_prune'):
                    self.pruner.step()
                self.optimizer.clear_grad()
                self.status['learning_rate'] = curr_lr

                if self._nranks < 2 or self._local_rank in self.log_ranks:
                    self.status['training_staus'].update(outputs)

                self.status['batch_time'].update(time.time() - iter_tic)
                self._compose_callback.on_step_end(self.status)
                if self.use_ema:
                    self.ema.update()
                iter_tic = time.time()

            if self.cfg.get('unstructured_prune'):
                self.pruner.update_params()

            is_snapshot = (self._nranks < 2 or (self._local_rank == 0 or self.cfg.metric == "Pose3DEval")) \
                       and ((epoch_id + 1) % self.cfg.snapshot_epoch == 0 or epoch_id == self.end_epoch - 1)
            if is_snapshot and self.use_ema:
                # apply ema weight on model
                weight = copy.deepcopy(self.model.state_dict())
                self.model.set_dict(self.ema.apply())
                self.status['weight'] = weight

            self._compose_callback.on_epoch_end(self.status)

            if validate and is_snapshot:
                if not hasattr(self, '_eval_loader'):
                    # build evaluation dataset and loader
                    self._eval_dataset = self.cfg.EvalDataset
                    self._eval_batch_sampler = \
                        paddle.io.BatchSampler(
                            self._eval_dataset,
                            batch_size=self.cfg.EvalReader['batch_size'])
                    # If metric is VOC, need to be set collate_batch=False.
                    if self.cfg.metric == 'VOC':
                        self.cfg['EvalReader']['collate_batch'] = False
                    if self.cfg.metric == "Pose3DEval":
                        self._eval_loader = create('EvalReader')(
                            self._eval_dataset, self.cfg.worker_num)
                    else:
                        self._eval_loader = create('EvalReader')(
                            self._eval_dataset,
                            self.cfg.worker_num,
                            batch_sampler=self._eval_batch_sampler)
                # if validation in training is enabled, metrics should be re-init
                # Init_mark makes sure this code will only execute once
                if validate and Init_mark == False:
                    Init_mark = True
                    self._init_metrics(validate=validate)
                    self._reset_metrics()

                with paddle.no_grad():
                    self.status['save_best_model'] = True
                    self._eval_with_loader(self._eval_loader)

            if is_snapshot and self.use_ema:
                # reset original weight
                self.model.set_dict(weight)
                self.status.pop('weight')

            #raise Exception("DEBUG")

        self._compose_callback.on_train_end(self.status)

    def _eval_with_loader(self, loader):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'

        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num, self._eval_batch_sampler)
            self._flops(flops_loader)
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            if self.use_amp:
                with paddle.amp.auto_cast(
                        enable=self.cfg.use_gpu or self.cfg.use_npu or
                        self.cfg.use_mlu,
                        custom_white_list=self.custom_white_list,
                        custom_black_list=self.custom_black_list,
                        level=self.amp_level):
                    outs = self.model(data)
            else:
                outs = self.model(data)

            # update metrics
            for metric in self._metrics:
                metric.update(data, outs)

            # multi-scale inputs: all inputs have same im_id
            if isinstance(data, typing.Sequence):
                sample_num += data[0]['im_id'].numpy().shape[0]
            else:
                sample_num += data['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        self._compose_callback.on_epoch_end(self.status)
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def evaluate(self):
        # get distributed model
        if self.cfg.get('fleet', False):
            self.model = fleet.distributed_model(self.model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg[
                'find_unused_parameters'] if 'find_unused_parameters' in self.cfg else False
            self.model = paddle.DataParallel(
                self.model, find_unused_parameters=find_unused_parameters)
        with paddle.no_grad():
            self._eval_with_loader(self.loader)

    def _eval_with_loader_slice(self,
                                loader,
                                slice_size=[640, 640],
                                overlap_ratio=[0.25, 0.25],
                                combine_method='nms',
                                match_threshold=0.6,
                                match_metric='iou'):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num, self._eval_batch_sampler)
            self._flops(flops_loader)

        merged_bboxs = []
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            if self.use_amp:
                with paddle.amp.auto_cast(
                        enable=self.cfg.use_gpu or self.cfg.use_npu or
                        self.cfg.use_mlu,
                        custom_white_list=self.custom_white_list,
                        custom_black_list=self.custom_black_list,
                        level=self.amp_level):
                    outs = self.model(data)
            else:
                outs = self.model(data)

            shift_amount = data['st_pix']
            outs['bbox'][:, 2:4] = outs['bbox'][:, 2:4] + shift_amount
            outs['bbox'][:, 4:6] = outs['bbox'][:, 4:6] + shift_amount
            merged_bboxs.append(outs['bbox'])

            if data['is_last'] > 0:
                # merge matching predictions
                merged_results = {'bbox': []}
                if combine_method == 'nms':
                    final_boxes = multiclass_nms(
                        np.concatenate(merged_bboxs), self.cfg.num_classes,
                        match_threshold, match_metric)
                    merged_results['bbox'] = np.concatenate(final_boxes)
                elif combine_method == 'concat':
                    merged_results['bbox'] = np.concatenate(merged_bboxs)
                else:
                    raise ValueError(
                        "Now only support 'nms' or 'concat' to fuse detection results."
                    )
                merged_results['im_id'] = np.array([[0]])
                merged_results['bbox_num'] = np.array(
                    [len(merged_results['bbox'])])

                merged_bboxs = []
                data['im_id'] = data['ori_im_id']
                # update metrics
                for metric in self._metrics:
                    metric.update(data, merged_results)

                # multi-scale inputs: all inputs have same im_id
                if isinstance(data, typing.Sequence):
                    sample_num += data[0]['im_id'].numpy().shape[0]
                else:
                    sample_num += data['im_id'].numpy().shape[0]

            self._compose_callback.on_step_end(self.status)

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic

        # accumulate metric to log out
        for metric in self._metrics:
            metric.accumulate()
            metric.log()
        self._compose_callback.on_epoch_end(self.status)
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def evaluate_slice(self,
                       slice_size=[640, 640],
                       overlap_ratio=[0.25, 0.25],
                       combine_method='nms',
                       match_threshold=0.6,
                       match_metric='iou'):
        with paddle.no_grad():
            self._eval_with_loader_slice(self.loader, slice_size, overlap_ratio,
                                         combine_method, match_threshold,
                                         match_metric)

    def slice_predict(self,
                      images,
                      slice_size=[640, 640],
                      overlap_ratio=[0.25, 0.25],
                      combine_method='nms',
                      match_threshold=0.6,
                      match_metric='iou',
                      draw_threshold=0.5,
                      output_dir='output',
                      save_results=False,
                      visualize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_slice_images(images, slice_size, overlap_ratio)
        loader = create('TestReader')(self.dataset, 0)
        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg else None
            output_eval = self.cfg[
                'output_eval'] if 'output_eval' in self.cfg else None

            # modify
            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file)

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)

        results = []  # all images
        merged_bboxs = []  # single image
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)

            outs['bbox'] = outs['bbox'].numpy()  # only in test mode
            shift_amount = data['st_pix']
            outs['bbox'][:, 2:4] = outs['bbox'][:, 2:4] + shift_amount.numpy()
            outs['bbox'][:, 4:6] = outs['bbox'][:, 4:6] + shift_amount.numpy()
            merged_bboxs.append(outs['bbox'])

            if data['is_last'] > 0:
                # merge matching predictions
                merged_results = {'bbox': []}
                if combine_method == 'nms':
                    final_boxes = multiclass_nms(
                        np.concatenate(merged_bboxs), self.cfg.num_classes,
                        match_threshold, match_metric)
                    merged_results['bbox'] = np.concatenate(final_boxes)
                elif combine_method == 'concat':
                    merged_results['bbox'] = np.concatenate(merged_bboxs)
                else:
                    raise ValueError(
                        "Now only support 'nms' or 'concat' to fuse detection results."
                    )
                merged_results['im_id'] = np.array([[0]])
                merged_results['bbox_num'] = np.array(
                    [len(merged_results['bbox'])])

                merged_bboxs = []
                data['im_id'] = data['ori_im_id']

                for _m in metrics:
                    _m.update(data, merged_results)

                for key in ['im_shape', 'scale_factor', 'im_id']:
                    if isinstance(data, typing.Sequence):
                        merged_results[key] = data[0][key]
                    else:
                        merged_results[key] = data[key]
                for key, value in merged_results.items():
                    if hasattr(value, 'numpy'):
                        merged_results[key] = value.numpy()
                results.append(merged_results)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid)
                bbox_num = outs['bbox_num']

                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    image_path = imid2path[int(im_id)]
                    image = Image.open(image_path).convert('RGB')
                    image = ImageOps.exif_transpose(image)
                    self.status['original_image'] = np.array(image.copy())

                    end = start + bbox_num[i]
                    bbox_res = batch_res['bbox'][start:end] \
                            if 'bbox' in batch_res else None
                    mask_res = batch_res['mask'][start:end] \
                            if 'mask' in batch_res else None
                    segm_res = batch_res['segm'][start:end] \
                            if 'segm' in batch_res else None
                    keypoint_res = batch_res['keypoint'][start:end] \
                            if 'keypoint' in batch_res else None
                    pose3d_res = batch_res['pose3d'][start:end] \
                            if 'pose3d' in batch_res else None
                    image = visualize_results(
                        image, bbox_res, mask_res, segm_res, keypoint_res,
                        pose3d_res, int(im_id), catid2name, draw_threshold)
                    self.status['result_image'] = np.array(image.copy())
                    if self._compose_callback:
                        self._compose_callback.on_step_end(self.status)
                    # save image with detection
                    save_name = self._get_save_image_name(output_dir,
                                                          image_path)
                    logger.info("Detection bbox results save in {}".format(
                        save_name))
                    image.save(save_name, quality=95)

                    start = end

    def predict(self,
                images,
                draw_threshold=0.5,
                output_dir='output',
                save_results=False,
                visualize=True,
                save_threshold=0,
                do_eval=False):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if do_eval:
            save_threshold = 0.0

        print(f"images inside predict: {images}")

        self.dataset.set_images(images, do_eval=do_eval)

        print(f"self.dataset: {type(self.dataset)}")
        #print(f"self.dataset: {self.dataset.__getitem__(0)}")
        print(f"create('TestReader'): {create('TestReader')}")

        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg else None
            output_eval = self.cfg[
                'output_eval'] if 'output_eval' in self.cfg else None

            # modify
            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self.cfg['save_threshold'] = save_threshold
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only            

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file)

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []
        results_2 = []
        results_final = []

        def transform_image(
            image: paddle.Tensor, 
            rotate_degrees: float = 0, 
            scale_factor: paddle.Tensor = paddle.to_tensor([1, 1]), 
            flip: str = None
        ) -> paddle.Tensor:
            """
            Applies scaling, rotation, and flipping transformations to an input image.
            
            Parameters:
                image (paddle.Tensor): Input image tensor of shape [1, C, H, W].
                rotate_degrees (float): Rotation angle (0, 90, 180, 270 degrees only).
                scale_factor (paddle.Tensor): Scaling factors [height_scale, width_scale] of size [1, 2].
                flip (str): Flip mode, one of {None, "horizontal", "vertical"}.
                
            Returns:
                paddle.Tensor: Transformed image tensor.
            """
            # Validate rotation angle
            if rotate_degrees not in [0, 90, 180, 270]:
                raise ValueError("rotate_degrees must be one of [0, 90, 180, 270].")
            
            # Extract scale factors
            scale_h, scale_w = scale_factor[0].item(), scale_factor[1].item()
            
            # Scaling
            height, width = image.shape[2], image.shape[3]
            scaled_height = int(height * scale_h)
            scaled_width = int(width * scale_w)
            transformed_image = F.interpolate(image, size=(scaled_height, scaled_width), mode='bilinear')
            
            # Rotation
            if rotate_degrees == 90:
                transformed_image = transformed_image.transpose([0, 1, 3, 2]).flip(axis=[2])  # Transpose and flip height
            elif rotate_degrees == 180:
                transformed_image = transformed_image.flip(axis=[2, 3])  # Flip both height and width
            elif rotate_degrees == 270:
                transformed_image = transformed_image.transpose([0, 1, 3, 2]).flip(axis=[3])  # Transpose and flip width
            
            # Flip
            if flip == "horizontal":
                transformed_image = transformed_image.flip(axis=[3])  # Flip horizontally
            elif flip == "vertical":
                transformed_image = transformed_image.flip(axis=[2])  # Flip vertically
            elif flip is not None:
                raise ValueError("flip must be one of {None, 'horizontal', 'vertical'}.")
            
            return transformed_image

        def ensemble_bounding_boxes(original_boxes, flipped_boxes, iou_threshold=0.5):
            """
            Ensembles bounding boxes from original and flipped images using Non-Maximum Suppression (NMS).
            
            Args:
                original_boxes (list of tuples): Bounding boxes from the original image in the format:
                    (class, x_center, y_center, width, height, confidence).
                flipped_boxes (list of tuples): Bounding boxes from the flipped image (mapped back to the original coordinates).
                iou_threshold (float): IoU threshold for merging overlapping boxes during NMS.

            Returns:
                list of tuples: Ensembled bounding boxes in the same format as input.
            """
            def convert_to_xyxy(box):
                """Convert (x_center, y_center, width, height) to (x_min, y_min, x_max, y_max)."""
                x_center, y_center, width, height = box
                x_min = x_center - width / 2
                y_min = y_center - height / 2
                x_max = x_center + width / 2
                y_max = y_center + height / 2
                return x_min, y_min, x_max, y_max

            def convert_to_xywh(box):
                """Convert (x_min, y_min, x_max, y_max) to (x_center, y_center, width, height)."""
                x_min, y_min, x_max, y_max = box
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                return x_center, y_center, width, height

            def iou(box1, box2):
                """Compute Intersection over Union (IoU) for two boxes."""
                x_min1, y_min1, x_max1, y_max1 = box1
                x_min2, y_min2, x_max2, y_max2 = box2

                # Compute the area of intersection
                inter_x_min = max(x_min1, x_min2)
                inter_y_min = max(y_min1, y_min2)
                inter_x_max = min(x_max1, x_max2)
                inter_y_max = min(y_max1, y_max2)

                inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

                # Compute the area of both boxes
                area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
                area2 = (x_max2 - x_min2) * (y_max2 - y_min2)

                # Compute IoU
                union_area = area1 + area2 - inter_area
                return inter_area / union_area if union_area > 0 else 0

            def nms(bboxes, iou_threshold):
                """Apply Non-Maximum Suppression (NMS) to merge overlapping boxes."""
                if not bboxes:
                    return []

                # Sort boxes by confidence score (descending)
                bboxes = sorted(bboxes, key=lambda x: x[-1], reverse=True)

                final_boxes = []
                while bboxes:
                    # Take the box with the highest confidence
                    best_box = bboxes.pop(0)
                    final_boxes.append(best_box)

                    # Remove boxes with high IoU overlap
                    bboxes = [
                        box for box in bboxes 
                        if iou(convert_to_xyxy(best_box[1:5]), convert_to_xyxy(box[1:5])) < iou_threshold
                    ]

                return final_boxes

            # Combine all boxes
            all_boxes = original_boxes + flipped_boxes

            # Group boxes by class
            class_groups = {}
            for box in all_boxes:
                class_label = box[0]
                if class_label not in class_groups:
                    class_groups[class_label] = []
                class_groups[class_label].append(box)

            # Apply NMS for each class
            ensembled_boxes = []
            for class_label, boxes in class_groups.items():
                merged_boxes = nms(boxes, iou_threshold=iou_threshold)
                ensembled_boxes.extend(merged_boxes)

            return ensembled_boxes

        def flip_back_bounding_boxes(bounding_boxes: np.ndarray, 
                                    rotate_degrees: float = 0, 
                                    scale_factor: paddle.Tensor = paddle.to_tensor([1, 1]), 
                                    flip: str = None) -> np.ndarray:
            """
            Flips back normalized bounding boxes to their original positions based on the inverse transformations.
            
            Parameters:
                bounding_boxes (np.ndarray): Array of shape [n, 5] where each row is 
                                            (class_id, x_center, y_center, width, height), normalized to [0, 1].
                rotate_degrees (float): Rotation angle (0, 90, 180, 270 degrees only).
                scale_factor (paddle.Tensor): Scaling factors [height_scale, width_scale] of size [1, 2]. Default is [1, 1].
                flip (str): Flip mode, one of {None, "horizontal", "vertical"}.
                image_size (tuple): Original image size as (height, width).
                
            Returns:
                np.ndarray: Transformed bounding boxes of shape [n, 5].
            """
            # height, width = image_size
            scale_h, scale_w = scale_factor[0].item(), scale_factor[1].item()
            
            # Reverse scaling
            bounding_boxes[:, 1] *= scale_w  # Reverse x_center scaling
            bounding_boxes[:, 2] *= scale_h  # Reverse y_center scaling
            bounding_boxes[:, 3] *= scale_w  # Reverse width scaling
            bounding_boxes[:, 4] *= scale_h  # Reverse height scaling
            
            # Reverse flipping
            if flip == "horizontal":
                bounding_boxes[:, 1] = 1 - bounding_boxes[:, 1]  # Reverse horizontal flip
            elif flip == "vertical":
                bounding_boxes[:, 2] = 1 - bounding_boxes[:, 2]  # Reverse vertical flip

            # Reverse rotation
            if rotate_degrees == 90:
                new_x = bounding_boxes[:, 2]
                new_y = 1 - bounding_boxes[:, 1]
                bounding_boxes[:, 1] = new_x
                bounding_boxes[:, 2] = new_y
            elif rotate_degrees == 180:
                bounding_boxes[:, 1] = 1 - bounding_boxes[:, 1]
                bounding_boxes[:, 2] = 1 - bounding_boxes[:, 2]
            elif rotate_degrees == 270:
                new_x = 1 - bounding_boxes[:, 2]
                new_y = bounding_boxes[:, 1]
                bounding_boxes[:, 1] = new_x
                bounding_boxes[:, 2] = new_y

            return bounding_boxes


        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            
            data_flipped = copy.deepcopy(data)
            data_flipped["image"] = transform_image(data["image"], flip="horizontal")
            if hasattr(self.model, 'modelTeacher'):
                #outs = self.model.modelTeacher(data)
                outs_2 = self.model.modelTeacher(data_flipped)
            else:
                #outs = self.model(data)
                outs_2 = self.model(data_flipped)

            outs_2["bbox"] = flip_back_bounding_boxes(outs_2["bbox"], flip="horizontal")

            print("data: ...")
            for key, item in data.items():
                print(f"{key}: {item.shape}, {type(item)}")
                if key == "scale_factor":
                    print(item)

            for _m in metrics:
                _m.update(data, outs)
                _m.update(data_flipped, outs_2)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data_flipped, typing.Sequence):
                    outs_2[key] = data_flipped[0][key]
                else:
                    outs_2[key] = data_flipped[key]
            for key, value in outs_2.items():
                if hasattr(value, 'numpy'):
                    outs_2[key] = value.numpy()

            print("outs: ...")
            for key, item in outs_2.items():
                print(f"{key}: {item.shape}, {type(item)}")  
            
            #if step_id > 0:
            #    raise Exception("DEBUG")

            #outs_final = copy.deepcopy(outs)
            #outs_final["bbox"] = ensemble_bounding_boxes(
            #    outs["bbox"], outs_2["bbox"]
            #)

            #results.append(outs)
            results_2.append(outs_2)
            #results_final.append(outs_final)

        # sniper
        if type(self.dataset) == SniperCOCODataSet:
            results = self.dataset.anno_cropper.aggregate_chips_detections(
                results)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            print("START INFERRING...")
            bbox_finals = []
            img_id_finals = []
            for outs in results_2:
                batch_res = get_infer_results(outs, clsid2catid)
                bbox_num = outs['bbox_num']

                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    image_path = imid2path[int(im_id)]
                    image = Image.open(image_path).convert('RGB')
                    image = ImageOps.exif_transpose(image)

                    #image = image.transpose(Image.FLIP_LEFT_RIGHT)

                    self.status['original_image'] = np.array(image.copy())

                    end = start + bbox_num[i]
                    bbox_res = batch_res['bbox'][start:end] \
                            if 'bbox' in batch_res else None
                    mask_res = batch_res['mask'][start:end] \
                            if 'mask' in batch_res else None
                    segm_res = batch_res['segm'][start:end] \
                            if 'segm' in batch_res else None
                    keypoint_res = batch_res['keypoint'][start:end] \
                            if 'keypoint' in batch_res else None
                    pose3d_res = batch_res['pose3d'][start:end] \
                            if 'pose3d' in batch_res else None
                    image = visualize_results(
                        image, bbox_res, mask_res, segm_res, keypoint_res,
                        pose3d_res, int(im_id), catid2name, draw_threshold)
                    print(bbox_res, mask_res, segm_res, keypoint_res,
                        pose3d_res, int(im_id), catid2name, draw_threshold, sep="\n")
                    bbox_finals.extend(bbox_res)
                    self.status['result_image'] = np.array(image.copy())
                    if self._compose_callback:
                        self._compose_callback.on_step_end(self.status)
                    # save image with detection
                    save_name = self._get_save_image_name(output_dir,
                                                          image_path)
                    logger.info("Detection bbox results save in {}".format(
                        save_name))
                    image.save(save_name, quality=95)

                    raise ValueError("DEBUG")

                    start = end

                    img_id_finals.append({
                        "image_path": save_name.split("/")[-1],
                        "image_id": bbox_res[0]["image_id"],
                        "image_height": image.height,
                        "image_width": image.width,
                    })
                    print(image.height, image.width)

                    # raise ValueError("DEBUG")
                    
                    # break
            
            with open("output/infer_annotations.json", "w") as f:
                json.dump(bbox_finals, f, indent=4)
            with open("output/infer_paths.json", "w") as f:
                json.dump(img_id_finals, f, indent=4)
        return results

    def _get_save_image_name(self, output_dir, image_path):
        """
        Get save image name from source image path.
        """
        image_name = os.path.split(image_path)[-1]
        name, ext = os.path.splitext(image_name)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        return os.path.join(output_dir, "images", "{}".format(name)) + ext

    def _get_infer_cfg_and_input_spec(self,
                                      save_dir,
                                      prune_input=True,
                                      kl_quant=False,
                                      yaml_name=None,
                                      model=None):
        if yaml_name is None:
            yaml_name = 'infer_cfg.yml'
        if model is None:
            model = self.model
        image_shape = None
        im_shape = [None, 2]
        scale_factor = [None, 2]
        if self.cfg.architecture in MOT_ARCH:
            test_reader_name = 'TestMOTReader'
        else:
            test_reader_name = 'TestReader'
        if 'inputs_def' in self.cfg[test_reader_name]:
            inputs_def = self.cfg[test_reader_name]['inputs_def']
            image_shape = inputs_def.get('image_shape', None)
        # set image_shape=[None, 3, -1, -1] as default
        if image_shape is None:
            image_shape = [None, 3, -1, -1]

        if len(image_shape) == 3:
            image_shape = [None] + image_shape
        else:
            im_shape = [image_shape[0], 2]
            scale_factor = [image_shape[0], 2]

        if hasattr(model, 'deploy'):
            model.deploy = True
        if 'slim' not in self.cfg:
            for layer in model.sublayers():
                if hasattr(layer, 'convert_to_deploy'):
                    layer.convert_to_deploy()

        if hasattr(self.cfg, 'export') and 'fuse_conv_bn' in self.cfg[
                'export'] and self.cfg['export']['fuse_conv_bn']:
            model = fuse_conv_bn(model)

        export_post_process = self.cfg['export'].get(
            'post_process', False) if hasattr(self.cfg, 'export') else True
        export_nms = self.cfg['export'].get('nms', False) if hasattr(
            self.cfg, 'export') else True
        export_benchmark = self.cfg['export'].get(
            'benchmark', False) if hasattr(self.cfg, 'export') else False
        if hasattr(model, 'fuse_norm'):
            model.fuse_norm = self.cfg['TestReader'].get('fuse_normalize',
                                                              False)
        if hasattr(model, 'export_post_process'):
            model.export_post_process = export_post_process if not export_benchmark else False
        if hasattr(model, 'export_nms'):
            model.export_nms = export_nms if not export_benchmark else False
        if export_post_process and not export_benchmark:
            image_shape = [None] + image_shape[1:]

        # Save infer cfg
        _dump_infer_config(self.cfg,
                           os.path.join(save_dir, yaml_name), image_shape,
                           model)

        input_spec = [{
            "image": InputSpec(
                shape=image_shape, name='image'),
            "im_shape": InputSpec(
                shape=im_shape, name='im_shape'),
            "scale_factor": InputSpec(
                shape=scale_factor, name='scale_factor')
        }]
        if self.cfg.architecture == 'DeepSORT':
            input_spec[0].update({
                "crops": InputSpec(
                    shape=[None, 3, 192, 64], name='crops')
            })

        if self.cfg.architecture == 'CLRNet':
            input_spec[0].update({
                "full_img_path": str,
                "img_name": str,
            })
        if prune_input:
            static_model = paddle.jit.to_static(
                model, input_spec=input_spec, full_graph=True)
            # NOTE: dy2st do not pruned program, but jit.save will prune program
            # input spec, prune input spec here and save with pruned input spec
            pruned_input_spec = _prune_input_spec(
                input_spec, static_model.forward.main_program,
                static_model.forward.outputs)
        else:
            static_model = None
            pruned_input_spec = input_spec

        # TODO: Hard code, delete it when support prune input_spec.
        if self.cfg.architecture == 'PicoDet' and not export_post_process:
            pruned_input_spec = [{
                "image": InputSpec(
                    shape=image_shape, name='image')
            }]
        if kl_quant:
            if self.cfg.architecture == 'PicoDet' or 'ppyoloe' in self.cfg.weights:
                pruned_input_spec = [{
                    "image": InputSpec(
                        shape=image_shape, name='image'),
                    "scale_factor": InputSpec(
                        shape=scale_factor, name='scale_factor')
                }]
            elif 'tinypose' in self.cfg.weights:
                pruned_input_spec = [{
                    "image": InputSpec(
                        shape=image_shape, name='image')
                }]

        return static_model, pruned_input_spec

    def export(self, output_dir='output_inference', for_fd=False):
        if hasattr(self.model, 'aux_neck'):
            self.model.__delattr__('aux_neck')
        if hasattr(self.model, 'aux_head'):
            self.model.__delattr__('aux_head')
        self.model.eval()
        model = copy.deepcopy(self.model)

        model_name = os.path.splitext(os.path.split(self.cfg.filename)[-1])[0]
        if for_fd:
            save_dir = output_dir
            save_name = 'inference'
            yaml_name = 'inference.yml'
        else:
            save_dir = os.path.join(output_dir, model_name)
            save_name = 'model'
            yaml_name = None

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        static_model, pruned_input_spec = self._get_infer_cfg_and_input_spec(
            save_dir, yaml_name=yaml_name, model=model)

        # dy2st and save model
        if 'slim' not in self.cfg or 'QAT' not in self.cfg['slim_type']:
            paddle.jit.save(
                static_model,
                os.path.join(save_dir, save_name),
                input_spec=pruned_input_spec)
        else:
            self.cfg.slim.save_quantized_model(
                self.model,
                os.path.join(save_dir, save_name),
                input_spec=pruned_input_spec)
        logger.info("Export model and saved in {}".format(save_dir))

    def post_quant(self, output_dir='output_inference'):
        model_name = os.path.splitext(os.path.split(self.cfg.filename)[-1])[0]
        save_dir = os.path.join(output_dir, model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for idx, data in enumerate(self.loader):
            self.model(data)
            if idx == int(self.cfg.get('quant_batch_num', 10)):
                break

        # TODO: support prune input_spec
        kl_quant = True if hasattr(self.cfg.slim, 'ptq') else False
        _, pruned_input_spec = self._get_infer_cfg_and_input_spec(
            save_dir, prune_input=False, kl_quant=kl_quant)

        self.cfg.slim.save_quantized_model(
            self.model,
            os.path.join(save_dir, 'model'),
            input_spec=pruned_input_spec)
        logger.info("Export Post-Quant model and saved in {}".format(save_dir))

    def _flops(self, loader):
        if hasattr(self.model, 'aux_neck'):
            self.model.__delattr__('aux_neck')
        if hasattr(self.model, 'aux_head'):
            self.model.__delattr__('aux_head')
        self.model.eval()
        try:
            import paddleslim
        except Exception as e:
            logger.warning(
                'Unable to calculate flops, please install paddleslim, for example: `pip install paddleslim`'
            )
            return

        from paddleslim.analysis import dygraph_flops as flops
        input_data = None
        for data in loader:
            input_data = data
            break

        input_spec = [{
            "image": input_data['image'][0].unsqueeze(0),
            "im_shape": input_data['im_shape'][0].unsqueeze(0),
            "scale_factor": input_data['scale_factor'][0].unsqueeze(0)
        }]
        flops = flops(self.model, input_spec) / (1000**3)
        logger.info(" Model FLOPs : {:.6f}G. (image shape is {})".format(
            flops, input_data['image'][0].unsqueeze(0).shape))

    def parse_mot_images(self, cfg):
        import glob
        # for quant
        dataset_dir = cfg['EvalMOTDataset'].dataset_dir
        data_root = cfg['EvalMOTDataset'].data_root
        data_root = '{}/{}'.format(dataset_dir, data_root)
        seqs = os.listdir(data_root)
        seqs.sort()
        all_images = []
        for seq in seqs:
            infer_dir = os.path.join(data_root, seq)
            assert infer_dir is None or os.path.isdir(infer_dir), \
                "{} is not a directory".format(infer_dir)
            images = set()
            exts = ['jpg', 'jpeg', 'png', 'bmp']
            exts += [ext.upper() for ext in exts]
            for ext in exts:
                images.update(glob.glob('{}/*.{}'.format(infer_dir, ext)))
            images = list(images)
            images.sort()
            assert len(images) > 0, "no image found in {}".format(infer_dir)
            all_images.extend(images)
            logger.info("Found {} inference images in total.".format(
                len(images)))
        return all_images

    def predict_culane(self,
                       images,
                       output_dir='output',
                       save_results=False,
                       visualize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg else None
            output_eval = self.cfg[
                'output_eval'] if 'output_eval' in self.cfg else None

            # modify
            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics

        if save_results:
            metrics = setup_metrics_for_loader()
        else:
            metrics = []

        # Run Infer
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []
        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)

            for _m in metrics:
                _m.update(data, outs)

            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            results.append(outs)

        for _m in metrics:
            _m.accumulate()
            _m.reset()

        if visualize:
            import cv2

            for outs in results:
                for i in range(len(outs['img_path'])):
                    lanes = outs['lanes'][i]
                    img_path = outs['img_path'][i]
                    img = cv2.imread(img_path)
                    out_file = os.path.join(output_dir,
                                            os.path.basename(img_path))
                    lanes = [
                        lane.to_array(
                            sample_y_range=[
                                self.cfg['sample_y']['start'],
                                self.cfg['sample_y']['end'],
                                self.cfg['sample_y']['step']
                            ],
                            img_w=self.cfg.ori_img_w,
                            img_h=self.cfg.ori_img_h) for lane in lanes
                    ]
                    imshow_lanes(img, lanes, out_file=out_file)

        return results

    def reset_norm_param_attr(self, layer, **kwargs):
        if isinstance(layer, (nn.BatchNorm2D, nn.LayerNorm, nn.GroupNorm)):
            src_state_dict = layer.state_dict()
            if isinstance(layer, nn.BatchNorm2D):
                layer = nn.BatchNorm2D(
                    num_features=layer._num_features,
                    momentum=layer._momentum,
                    epsilon=layer._epsilon,
                    **kwargs)
            elif isinstance(layer, nn.LayerNorm):
                layer = nn.LayerNorm(
                    normalized_shape=layer._normalized_shape,
                    epsilon=layer._epsilon,
                    **kwargs)
            else:
                layer = nn.GroupNorm(
                    num_groups=layer._num_groups,
                    num_channels=layer._num_channels,
                    epsilon=layer._epsilon,
                    **kwargs)
            layer.set_state_dict(src_state_dict)
        else:
            for name, sublayer in layer.named_children():
                new_sublayer = self.reset_norm_param_attr(sublayer, **kwargs)
                if new_sublayer is not sublayer:
                    setattr(layer, name, new_sublayer)

        return layer
