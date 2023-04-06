import argparse
import sys
import time
import colorsys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from modeling.rdd_detector import create_detectorRDD
from loss.proprocess import process_prediction, location_to_box, soft_nms
from utils.utils import generate_grid2D


def resize_pad_img(image, target_shape, fill):
    '''
    image resize.
    '''
    # get original image shape
    img_shape = tf.shape(image)[0:2]
    # get padding offset
    scale = tf.reduce_min(target_shape / img_shape)
    scale = tf.cast(scale, dtype=tf.float32)
    aspect_ratio = tf.cast(img_shape, dtype=tf.float32) * scale
    aspect_ratio = tf.cast(aspect_ratio, tf.int32)
    image = tf.image.resize(image, aspect_ratio)  # method=ResizeMethod.BILINEAR

    padding = target_shape - aspect_ratio
    padding_lt = padding // 2
    padding_rb = padding - padding_lt
    image = tf.pad(image, [[padding_lt[0], padding_rb[0]], [padding_lt[1], padding_rb[1]], [0, 0]], constant_values=fill)
    # tf.pad(img[s_y1:s_y2, s_x1:s_x2], [[l_y1, 0], [l_x1, 0], [0, 0]], constant_values=144.0/255.0)
    return image, scale, padding_lt


class CModelRunning():
    def __init__(self, input_shape, model_name, weight_path):
        self.input_shape = input_shape
        self.score_threshold = .2
        self.iou_threshold = .6
        # self.createColors('annotation/voc_classes.txt')
        # self.createColors('annotation/coco_classes.txt')
        self.createColors('annotation/rdd_classes20.txt')
        self.num_classes = len(self.class_names)
        
        stride = [32, 16, 8]
        self.lstGrid = []
        for i in range(len(stride)):
            h = input_shape[0]//stride[i]
            w = input_shape[1]//stride[i]
            grid = generate_grid2D(h, w)
            self.lstGrid.append(tf.cast(grid, tf.float32))
        # creat model
        self.model_body = self.createmodel(input_shape, model_name, self.num_classes, weight_path)

    def createColors(self, category_path):
        """
        """
        with open(category_path) as f:
            category_lines = f.readlines()
        self.class_names = []
        for line in category_lines:
            n = line.find(',')
            e = line.find('\n')
            if n >= 0:
                self.class_names.append(line[n+1:e])
            else:
                self.class_names.append(line[:e])
        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

    def createmodel(self, input_shape, model_name, num_classes, weight_path):
        # model_body
        model_body, blockoutputs = create_detectorRDD(input_shape, num_classes, train=True, deepsupervise=True, weight_decay=0.0005)
        model_body.load_weights(weight_path)
        return model_body

    def detect_img(self, org_image):
        # read image
        image_data = np.array(org_image)
        image_data = tf.convert_to_tensor(image_data)
        # convert to float32 [0, 1.]
        image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)
        img_shape = tf.shape(image_data)[:2]
        img_shape = tf.cast(img_shape, tf.float32)
        img_shape = img_shape[::-1]
        image_data, scale, padding_lt = resize_pad_img(image_data, self.input_shape, 114/255.0)
        padding_lt = tf.cast(padding_lt[::-1], tf.float32)
        print('img_shape:', img_shape, 'scale:', scale, 'padding_lt:', padding_lt)
        # prediction
        image_data = tf.expand_dims(image_data, 0)
        model_out = self.model_body.predict_on_batch(image_data)
        # pro_process
        image_pred = [model_out[0][0], model_out[1][0], model_out[2][0]]
        pred_bboxes, all_cls_score = location_to_box(image_pred, self.lstGrid, self.input_shape, self.score_threshold)
        # print(all_cls_score)
        # prediction: list of np.array (6). l, t, r, b, c, score.
        # count_bboxes, out_bboxes = soft_nms(pred_bboxes, all_cls_score, self.score_threshold, self.iou_threshold, self.num_classes)
        # if count_bboxes > 0:
        out_bboxes = process_prediction(pred_bboxes, all_cls_score, self.score_threshold, self.iou_threshold, self.num_classes)
        if len(out_bboxes) > 0:
            out_bboxes = tf.concat(out_bboxes, axis=0)
            return self.drawResult(out_bboxes, org_image, img_shape, padding_lt, scale)
        else:
            return org_image

    def drawResult(self, out_bboxes, org_image, img_shape, padding_lt, scale):
        # for N boxes
        print('boxes:', out_bboxes)
        # colors = ['red', 'green']
        # color_i = 0
        # true_box = tf.convert_to_tensor([179, 63, 425, 478], dtype=tf.int32)
        # true_area = (425 - 179) * (478 - 63)
        # true_box = tf.convert_to_tensor([37, 107, 640, 449], dtype=tf.int32)
        # true_area = (449 - 107) * (640 - 37)
        draw = ImageDraw.Draw(org_image)
        # thickness = (org_image.width + org_image.height) // 300
        thickness = 2
        for n in range(len(out_bboxes)):
            boxes = out_bboxes[n]
            bboxes_lt = boxes[:2] * self.input_shape - padding_lt
            bboxes_rb = boxes[2:4] * self.input_shape - padding_lt
            bboxes_lt = bboxes_lt / scale
            bboxes_rb = bboxes_rb / scale
            bboxes_lt = tf.cast(bboxes_lt+.5, tf.int32)
            bboxes_rb = tf.cast(bboxes_rb+.5, tf.int32)
            bboxes_lt = tf.where(bboxes_lt < 0, 0, bboxes_lt)

            clsId = tf.cast(boxes[4], tf.int32)
            label = '{} {:.2f}'.format(self.class_names[clsId], boxes[5])
            # label = 'score:{0:.2f} iou:{1:.2f}'.format(boxes[5], iou)
            # print('label:', label)
            # 绘制类名
            font = ImageFont.truetype("consola.ttf", 24, encoding="unic")  # 设置字体
            draw.text(bboxes_lt+[0, -25], label, font=font)

            # fill=None,
            # label = '{} {:.2f}'.format(self.class_names[clsId], boxes[5])
            # print(label)
            # strPrediction = '{}:{},{},{},{}\n'.format(self.class_names[key], bboxes_x[i, 0], bboxes_y[i, 0], bboxes_x[i, 1], bboxes_y[i, 1])
            # print(strPrediction)
            for o in range(thickness):
                draw.rectangle([bboxes_lt[0] - o, bboxes_lt[1] - o, bboxes_rb[0] + o, bboxes_rb[1] + o], outline=self.colors[clsId], width=4)
            # color_i += 1
            # draw.rectangle([true_box[0], true_box[1], true_box[2], true_box[3]], outline='yellow', width=4)
        return org_image


def main(weight_path, model):
    input_shape = (704, 704)
    running = CModelRunning(input_shape, model, weight_path)
    while True:
        img_path = input('Input image filename:')
        try:
            if img_path == 'e':
                break
            image = Image.open(img_path)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = running.detect_img(image)
            r_image.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, help='model weights')
    parser.add_argument('--model', type=str, help='model type')

    args = parser.parse_args(sys.argv[1:])
    main(args.weights, args.model)
# python detect_image.py --model GRDDC2020 --weights weights/ep180-loss5.536.h5
