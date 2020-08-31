import argparse
import logging
import sys
import time
import os
import os.path as osp

from tf_pose import common
import cv2
import numpy as np
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ['CUDA_VISBLE_DEVICES']= '2'

def _gauss_on(img, center):
  #h, w = 11, 11
  #R = 0.5 * np.sqrt(h**2 + w**2)
  #gauss_map = np.zeros([11, 11])

  #for i in range(h):
  #  for j in range(w):
  #      dis = np.sqrt((i-h/2.0)**2.0+(j-w/2.0)**2.0)
  #      gauss_map[i, j] = np.exp(-0.5*dis/R)
  #gauss_map = gauss_map * 255.0
  #
  ver_begin = center[1] - 1 - 5
  ver_end = center[1] - 1 + 6
  hor_begin = center[0] - 1 - 5
  hor_end = center[0] - 1 + 6

  h, w = 5, 5
  #R = 0.5 * np.sqrt(h**2 + w**2)
  R = 2
  for i in range(ver_begin, ver_end):
    for j in range(hor_begin, hor_end):
      dist = int(np.sqrt((i-center[1])**2.0+(j-center[0])**2.0))
      if dist > R:
        continue
      else:
        if i >= img.shape[0] or j >= img.shape[1]:
          continue
        img[i, j] = np.exp(-0.5*dist/R) * 255.0
  
      


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    #parser.add_argument('--image', type=str, default='./images/p1.jpg')
    parser.add_argument('--base_path', type=str, default='')
    parser.add_argument('--model', type=str, default='cmu', help='cmu / mobilenet_thin')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')
    args = parser.parse_args()

    w, h = model_wh(args.resize)
    if w == 0 or h == 0:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))
    else:
        e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    for cur_location, folders, files in os.walk(args.base_path):
        if 'img_00001.jpg' not in files:
            continue
        else:
            for ind, img in enumerate(files):
                if img.startswith('flow'):
                  print('pass %s ' % img)
                  continue
                path_to_img = osp.join(cur_location, img)
                save_folder = osp.join('/data1/zhiyuyin/experiments/ICML/UCSD-pose', cur_location.split(os.sep)[-2], cur_location.split(os.sep)[-1])
                if not osp.exists(save_folder):
                  os.makedirs(save_folder)
                path_to_save = osp.join(save_folder, 'pose_' + img[4:9] + '.npy')


                # estimate human poses from a single image !
                image = common.read_imgfile(path_to_img, None, None)
                if image is None:
                    logger.error('Image can not be read, path=%s' % args.image)
                    sys.exit(-1)
                t = time.time()
                humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
                elapsed = time.time() - t
                logger.info('inference image: %s in %.4f seconds.' % (path_to_img, elapsed))
 

          
 
                from tf_pose import common 
                image_h, image_w = image.shape[:2]
                centers = {}
                pose_channels = [np.zeros([image_h, image_w]) for i in range(common.CocoPart.Background.value)]

                if len(humans) <= 0 and osp.basename(cur_location).startswith('Train'):
                    continue


                for human in humans:
                    if len(humans) <= 0:
                        break
                    # draw point
                    for i in range(common.CocoPart.Background.value):
                        if i not in human.body_parts.keys():
                            continue 
                        body_part = human.body_parts[i]
                        center = (int(body_part.x * image_w + 0.5), int(body_part.y *image_h + 0.5))
                        centers[i] = center
                        
                        _gauss_on(pose_channels[i], center)


                pose_channels = [np.expand_dims(item, axis=-1) for item in pose_channels]
                pose_channels = np.concatenate(pose_channels, axis=-1)
                np.save(path_to_save, pose_channels)
                
