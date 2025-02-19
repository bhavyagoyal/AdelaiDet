# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.utils.events import EventStorage
from detectron2.evaluation import (
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.utils.logger import setup_logger

from adet.data.dataset_mapper import DatasetMapperWithBasis
from adet.config import get_cfg
from adet.checkpoint import AdetCheckpointer
from adet.evaluation import TextEvaluator
import copy
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data import MetadataCatalog, DatasetCatalog

class Trainer(DefaultTrainer):
    """
    This is the same Trainer except that we rewrite the
    `build_train_loader`/`resume_or_load` method.
    """
    def build_hooks(self):
        """
        Replace `DetectionCheckpointer` with `AdetCheckpointer`.

        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.
        """
        ret = super().build_hooks()
        for i in range(len(ret)):
            if isinstance(ret[i], hooks.PeriodicCheckpointer):
                self.checkpointer = AdetCheckpointer(
                    self.model,
                    self.cfg.OUTPUT_DIR,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                )
                ret[i] = hooks.PeriodicCheckpointer(self.checkpointer, self.cfg.SOLVER.CHECKPOINT_PERIOD)
        return ret
    
    def resume_or_load(self, resume=True):
        checkpoint = self.checkpointer.resume_or_load(self.cfg.MODEL.WEIGHTS, resume=resume)
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger("adet.trainer")
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            self.before_train()
            for self.iter in range(start_iter, max_iter):
                self.before_step()
                self.run_step()
                self.after_step()
            self.after_train()

    def train(self):
        """
        Run training.

        Returns:
            OrderedDict of results, if evaluation is enabled. Otherwise None.
        """
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Returns:
            iterable

        It calls :func:`detectron2.data.build_detection_train_loader` with a customized
        DatasetMapper, which adds categorical labels as a semantic mask.
        """
        mapper = DatasetMapperWithBasis(cfg, True)
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each builtin dataset.
        For your own dataset, you can simply create an evaluator manually in your
        script and do not have to worry about the hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
                    ignore_label=cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    output_dir=output_folder,
                )
            )
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if evaluator_type == "coco_panoptic_seg":
            evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, cfg, True, output_folder)
        if evaluator_type == "text":
            return TextEvaluator(dataset_name, cfg, True, output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        if len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("adet.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)

    rank = comm.get_rank()
    setup_logger(cfg.OUTPUT_DIR, distributed_rank=rank, name="adet")

    return cfg


def main(args):
    #city_loc = '/srv/home/bhavya/datasets/cityscapes/coco/'
    #city_loc = '/srv/home/bhavya/datasets/cityscapes_distortedbalance3/motion_shot/1/coco/'
    #city_loc = os.getenv("DETECTRON2_DATASETS", "datasets") + '/cityscapes/'
    #register_coco_instances("cityscapes_detection_train", {}, city_loc + 'annotations/instances_train2017.json', city_loc)
    #register_coco_instances("cityscapes_detection_val", {}, city_loc + 'annotations/instances_val2017.json', city_loc)


    if args.eval_only:
        args.opts.append('DATASETS.TRAIN')
        args.opts.append(('coco_2017_train_custom',))
        args.opts.append('DATASETS.TEST')
        args.opts.append(('coco_2017_val_custom',))
        #weights = ['_graymotionshot0', '_graymotionshot1', '_graymotionshot2', '_graymotionshot3', '_graymotionshot4', '_graymotionshot5']
        #locs = ['motion_shot/0', 'motion_shot/1', 'motion_shot/2', 'motion_shot/3', 'motion_shot/4', 'motion_shot/5']
        #weights = ['_graymotionshot1234', '_graymotionshot1234', '_graymotionshot1234', '_graymotionshot1234']
        locs = ['motion_shot/1', 'motion_shot/2', 'motion_shot/3', 'motion_shot/4']
        weights = ['_motionshot1', '_motionshot2', '_motionshot3', '_motionshot4']
        #locs = ['motion_shot/4', 'motion_shot/4']
        #weights = ['_motionshot4', '_motionshot4_repeat']
        ##weights = ['', '', '', '']
        ##locs = ['motion_shot/1', 'motion_shot/2', 'motion_shot/3', 'motion_shot/4']
        #weights = ['_motion12345', '_shot12345']
        #locs = ['motion_blur/5', 'shot_noise/5']
        ##weights = ['', '_repeat']
        ##weights = ['', '']
        #output_dir = cfg.OUTPUT_DIR
        output_dir = args.opts[-7]
        for i in range(len(weights)):
            #loc = '/srv/home/bhavya/datasets/coco17_distortedbalance3/' + locs[i] + '/coco/'
            #register_coco_instances("coco_2017_val_custom"+str(i), _get_builtin_metadata('coco'), loc + "annotations/instances_val2017.json", loc + "val2017")
            loc = '/srv/home/bhavya/datasets/cityscapes_distortedbalance3/' + locs[i] + '/cityscapes/'
            register_coco_instances("cityscapes_detection_val_custom"+str(i), {}, loc + "annotations/instances_val2017.json", loc)
            args.opts[-7] = output_dir + str(i)
            #args.opts[-5] = 'training_dir_balance3/fcos_R_50_1x' + weights[i] + '/model_final.pth'
            #args.opts[-1] = ('coco_2017_val_custom' + str(i),)
            args.opts[-5] = 'training_dir_cityscapes_balance3/fcos_R_50_1x' + weights[i] + '/model_final.pth'
            args.opts[-1] = ('cityscapes_detection_val_custom' + str(i),)
            cfg = setup(args)
            model = Trainer.build_model(cfg)
            model.proposal_generator.use_fcos_outputs = False
            model.proposal_generator.model_count = i
            AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model) # d2 defaults.py

        i=len(weights)-1
        #loc = '/srv/home/bhavya/datasets/coco17_distortedbalance3/' + locs[i] + '/coco/'
        #register_coco_instances("coco_2017_val_custom"+str(i+1), _get_builtin_metadata('coco'), loc + "annotations/instances_val2017.json", loc + "val2017")
        loc = '/srv/home/bhavya/datasets/cityscapes_distortedbalance3/' + locs[i] + '/cityscapes/'
        register_coco_instances("cityscapes_detection_val_custom"+str(i+1), {}, loc + "annotations/instances_val2017.json", loc)
        args.opts[-7] = output_dir
        #args.opts[-5] = 'training_dir_balance3/fcos_R_50_1x' + weights[i] + '/model_final.pth'
        #args.opts[-1] = ('coco_2017_val_custom' + str(i+1),)
        args.opts[-5] = 'training_dir_cityscapes_balance3/fcos_R_50_1x' + weights[i] + '/model_final.pth'
        args.opts[-1] = ('cityscapes_detection_val_custom' + str(i+1),)

        cfg = setup(args)
        model = Trainer.build_model(cfg)
        model.proposal_generator.use_fcos_outputs = True
        model.proposal_generator.model_count = i
        AdetCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model) # d2 defaults.py

        if comm.is_main_process():
            verify_results(cfg, res)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        return res

    cfg = setup(args)
    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop or subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
