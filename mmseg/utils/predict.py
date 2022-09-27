# Copyright (c) OpenMMLab. All rights reserved.
from ast import arg
from genericpath import isfile
import os
import glob


def ext(path):
    return os.path.splitext(path)[1]


def predict(config, checkpoint, img, out_file=None, device='cuda:0', palette='rust', opacity=0.5):
    from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
    from mmseg.core.evaluation import get_palette
    
    # make output folder
    if not out_file:
        out_dir = 'results'
    elif os.path.splitext(out_file)[1] == '':
        out_dir = out_file
    else:
        raise Exception('Sorry, output must be a folder')
    os.makedirs(out_dir, exist_ok=True)

    # check if the input from a single image file or from a folder
    img_exts = ['.jpg', '.JPG', '.jpeg', '.png']
    if  os.path.isdir(img):
        imgs_path = [img_path for img_path in glob.glob(f'{img}/**/*.*', recursive=True) if ext(img_path) in img_exts]
    elif os.path.isfile(img):
        imgs_path = [img]
    else:
        raise Exception('Sorry, input must be an image file or a folder')

    # get palette array
    palette_array = get_palette(palette) if isinstance(palette, str) else palette
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config, checkpoint, device=device)
    # predict
    for img_path in imgs_path:
        img_file = os.path.split(img_path)[1]
        result = inference_segmentor(model, img_path)
        # show the results
        show_result_pyplot(
            model,
            img_path,
            result,
            palette_array,
            opacity=opacity,
            out_file=os.path.join(out_dir, img_file))
