import glob
import os
import shutil
import random

import mmcv
import numpy as np
from PIL import Image

random.seed(3)
def split_train_val(dataset_path, img_dir="img_dir", anno_dir="anno_dir", ratio=0.8, force=True):
    imgs = os.listdir(os.path.join(dataset_path, img_dir))
    annos = os.listdir(os.path.join(dataset_path, anno_dir))
    if force:
        annos_name = [os.path.splitext(path)[0] for path in annos]
        imgs_copy = imgs[:]
        for img_file in imgs_copy:
            img_name = os.path.splitext(img_file)[0]
            if img_name not in annos_name:
                img_path = os.path.join(dataset_path, img_dir, img_file)
                os.remove(img_path)
                imgs.remove(img_file)
    else:
        assert len(imgs)==len(annos), f"num_imgs = {len(imgs)}, num_annos = {len(annos)}"

    for split_dir in ["train", "val"]:
        img_out_dir = os.path.join(dataset_path, img_dir, split_dir)
        anno_out_dir = os.path.join(dataset_path, anno_dir, split_dir)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(anno_out_dir, exist_ok=True)

    all_list = []
    imgs_path, annos_path = {}, {}
    for path in imgs:
        img_name = os.path.splitext(path)[0]
        anno = img_name + ".png"
        
        img_path = os.path.join(dataset_path, img_dir, path)
        anno_path = os.path.join(dataset_path, anno_dir, anno)
        imgs_path[img_name] = img_path
        annos_path[img_name] = anno_path
        all_list.append(img_name)
    
    random.shuffle(all_list)
    num_samples = len(all_list)
    num_train = int(num_samples*ratio)

    train_list = all_list[:num_train]
    val_list = all_list[num_train:]

    train_img_out_dir = os.path.join(dataset_path, img_dir, "train")
    train_anno_out_dir = os.path.join(dataset_path, anno_dir, "train")
    for img_name in train_list:
        img_path = imgs_path[img_name]
        anno_path = annos_path[img_name]
        shutil.move(img_path, train_img_out_dir)
        shutil.move(anno_path, train_anno_out_dir)
    
    val_img_out_dir = os.path.join(dataset_path, img_dir, "val")
    val_anno_out_dir = os.path.join(dataset_path, anno_dir, "val")
    for img_name in val_list:
        img_path = imgs_path[img_name]
        anno_path = annos_path[img_name]
        shutil.move(img_path, val_img_out_dir)
        shutil.move(anno_path, val_anno_out_dir)

def iSAID_convert_from_color(arr_3d, palette):
    """RGB-color encoding to grayscale labels."""
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d

def convert_ann(src_dir, dst_dir, palette, remove_suffix=True):
    annots = [annot for annot in sorted(glob.glob(f'{src_dir}/**/*.png', recursive=True))]
    for annot in annots:
        label = mmcv.imread(annot, channel_order='rgb')
        label = iSAID_convert_from_color(label, palette=palette)
        label = Image.fromarray(label.astype(np.uint8), mode='P')
        filepath = annot.split('/')
        filepath[-2] = dst_dir
        if remove_suffix:
            filepath[-1] = filepath[-1].replace("_mask", "")
        os.makedirs("/".join(filepath[:-1]), exist_ok=True)
        dst_path = "/".join(filepath)
        label.save(dst_path)