# Copyright (c) OpenMMLab. All rights reserved.
from ast import arg
from genericpath import isfile
import os
import glob
from argparse import ArgumentParser

from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette


def ext(path):
    return os.path.splitext(path)[1]


def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file/folder path')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output folder')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--palette',
        default='cityscapes',
        help='Color palette used for segmentation map')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    args = parser.parse_args()

    # make output folder
    if not args.out_file:
        out_dir = 'results'
    elif os.path.splitext(args.out_file)[1] == '':
        out_dir = args.out_file
    else:
        raise Exception('Sorry, output must be a folder')
    os.makedirs(out_dir, exist_ok=True)

    # check if the input from a single image file or from a folder
    img_exts = ['.jpg', '.JPG', '.jpeg', '.png']
    if  os.path.isdir(args.img):
        imgs_path = [img_path for img_path in glob.glob(f'{args.img}/**/*.*', recursive=True) if ext(img_path) in img_exts]
    elif os.path.isfile(args.img):
        imgs_path = [args.img]
    else:
        raise Exception('Sorry, input must be an image file or a folder')

    # build the model from a config file and a checkpoint file
    model = init_segmentor(args.config, args.checkpoint, device=args.device)
    # predict
    for img_path in imgs_path:
        img_file = os.path.split(img_path)[1]
        result = inference_segmentor(model, img_path)
        # show the results
        show_result_pyplot(
            model,
            img_path,
            result,
            get_palette(args.palette),
            opacity=args.opacity,
            out_file=os.path.join(out_dir, img_file))


if __name__ == '__main__':
    main()
