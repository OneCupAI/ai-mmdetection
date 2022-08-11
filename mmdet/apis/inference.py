# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv.ops import RoIPool
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from torch.nn.functional import interpolate, pad
from torchvision.transforms import Normalize

from mmdet.core import get_classes
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmdet.models import build_detector

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def determine_padding(sample_img, desiredDivisibility=32):
    numChannels, height, width = sample_img.shape

    heightPadding = height % desiredDivisibility
    widthPadding = width % desiredDivisibility

    if heightPadding != 0:
        heightPadding = desiredDivisibility - heightPadding

    if widthPadding != 0:
        widthPadding = desiredDivisibility - widthPadding

    return heightPadding, widthPadding


def init_detector(config, checkpoint=None, device='cuda:0', cfg_options=None):
    """Initialize a detector from config file.

    Args:
        config (str, :obj:`Path`, or :obj:`mmcv.Config`): Config file path,
            :obj:`Path`, or the config object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        cfg_options (dict): Options to override some settings in the used
            config.

    Returns:
        nn.Module: The constructed detector.
    """
    if isinstance(config, (str, Path)):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    if cfg_options is not None:
        config.merge_from_dict(cfg_options)
    if 'pretrained' in config.model:
        config.model.pretrained = None
    elif 'init_cfg' in config.model.backbone:
        config.model.backbone.init_cfg = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            warnings.simplefilter('once')
            warnings.warn('Class names are not saved in the checkpoint\'s '
                          'meta data, use COCO classes by default.')
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """Deprecated.

    A simple pipeline to load image.
    """

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.
        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        warnings.simplefilter('once')
        warnings.warn('`LoadImage` is deprecated and will be removed in '
                      'future releases. You may use `LoadImageFromWebcam` '
                      'from `mmdet.datasets.pipelines.` instead.')
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_fields'] = ['img']
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_detector(model, imgs):
    """Inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        imgs (str/ndarray or list[str/ndarray] or tuple[str/ndarray]):
           Either image files or loaded images.

    Returns:
        If imgs is a list or tuple, the same length list type results
        will be returned, otherwise return the detection results directly.
    """

    print('***************here1')

    normalize = Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])

    print('***************here2')

    # Extracting a sample frame (of 3 dimensions)
    sampleFrame = imgs[0, ...]

    print('***************here3')

    # Duplicating the shape to match the batch size (otherwise triton throws an error)
    ori_shape = imgs.shape[1:]

    print('***************here4')

    # Computing the scale factor
    print('***********in inference', sampleFrame.shape)
    _, scaleFactor = mmcv.imrescale(sampleFrame, (1333, 800), return_scale=True)

    # Converting to tensor
    imgs = torch.from_numpy(imgs).to(DEVICE)

    # Converting to RGB
    imgs = imgs.permute(0, 3, 1, 2)

    # Changing data type to float so the size can be interpolated
    imgs = imgs.type(torch.float)

    # Resizing the image
    imgs = interpolate(imgs, scale_factor=scaleFactor, mode='bilinear')
    scaleFactor = np.array([scaleFactor] * 4, dtype=np.float32)

    img_shape = np.array(list(imgs.shape[1:]))[[1, 2, 0]]

    # Determining the padding we want to add
    heightPadding, widthPadding = determine_padding(imgs[0, ...])

    # Adding the padding
    imgs = pad(imgs, (0, widthPadding, 0, heightPadding), value=0)

    imgs = normalize(imgs)
    #
    # print(scaleFactor.dtype)
    # print(ori_shape)
    # print(img_shape.dtype)

    img_metas = [[{'scale_factor': scaleFactor, 'ori_shape': ori_shape, 'img_shape': img_shape}]*imgs.shape[0]]
    data = {'img_metas': img_metas, 'img':[imgs]}

    # exit(0)



    # if isinstance(imgs, (list, tuple)):
    #     is_batch = True
    # else:
    #     imgs = [imgs]
    #     is_batch = False
    #
    # cfg = model.cfg
    # device = next(model.parameters()).device  # model device
    #
    # if isinstance(imgs[0], np.ndarray):
    #     cfg = cfg.copy()
    #     # set loading pipeline type
    #     cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    #
    # cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    # test_pipeline = Compose(cfg.data.test.pipeline)
    #
    # datas = []
    # for img in imgs:
    #     # prepare data
    #     if isinstance(img, np.ndarray):
    #         # directly add img
    #         data = dict(img=img)
    #     else:
    #         # add information into dict
    #         data = dict(img_info=dict(filename=img), img_prefix=None)
    #     # build the data pipeline
    #     data = test_pipeline(data)
    #     datas.append(data)
    #
    # data = collate(datas, samples_per_gpu=len(imgs))
    # # just get the actual data from DataContainer
    # data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    # data['img'] = [img.data[0] for img in data['img']]
    # if next(model.parameters()).is_cuda:
    #     # scatter to specified GPU
    #     data = scatter(data, [device])[0]
    # else:
    #     for m in model.modules():
    #         assert not isinstance(
    #             m, RoIPool
    #         ), 'CPU inference with RoIPool is not supported currently.'
    #
    # print(data['img'][0].dtype)

    # exit(0)
    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)

    return results


async def async_inference_detector(model, imgs):
    """Async inference image(s) with the detector.

    Args:
        model (nn.Module): The loaded detector.
        img (str | ndarray): Either image files or loaded images.

    Returns:
        Awaitable detection results.
    """
    if not isinstance(imgs, (list, tuple)):
        imgs = [imgs]

    cfg = model.cfg
    device = next(model.parameters()).device  # model device

    if isinstance(imgs[0], np.ndarray):
        cfg = cfg.copy()
        # set loading pipeline type
        cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'

    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    datas = []
    for img in imgs:
        # prepare data
        if isinstance(img, np.ndarray):
            # directly add img
            data = dict(img=img)
        else:
            # add information into dict
            data = dict(img_info=dict(filename=img), img_prefix=None)
        # build the data pipeline
        data = test_pipeline(data)
        datas.append(data)

    data = collate(datas, samples_per_gpu=len(imgs))
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
    else:
        for m in model.modules():
            assert not isinstance(
                m, RoIPool
            ), 'CPU inference with RoIPool is not supported currently.'

    # We don't restore `torch.is_grad_enabled()` value during concurrent
    # inference since execution can overlap
    torch.set_grad_enabled(False)
    results = await model.aforward_test(rescale=True, **data)
    return results


def show_result_pyplot(model,
                       img,
                       result,
                       score_thr=0.3,
                       title='result',
                       wait_time=0,
                       palette=None,
                       out_file=None):
    """Visualize the detection results on the image.

    Args:
        model (nn.Module): The loaded detector.
        img (str or np.ndarray): Image filename or loaded image.
        result (tuple[list] or list): The detection result, can be either
            (bbox, segm) or just bbox.
        score_thr (float): The threshold to visualize the bboxes and masks.
        title (str): Title of the pyplot figure.
        wait_time (float): Value of waitKey param. Default: 0.
        palette (str or tuple(int) or :obj:`Color`): Color.
            The tuple of color should be in BGR order.
        out_file (str or None): The path to write the image.
            Default: None.
    """
    if hasattr(model, 'module'):
        model = model.module
    model.show_result(
        img,
        result,
        score_thr=score_thr,
        show=True,
        wait_time=wait_time,
        win_name=title,
        bbox_color=palette,
        text_color=(200, 200, 200),
        mask_color=palette,
        out_file=out_file)
