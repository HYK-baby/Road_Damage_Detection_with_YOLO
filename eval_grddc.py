import argparse
import sys
import time
import numpy as np
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
from modeling.rdd_detector import create_detectorRDD
from utils.utils import generate_grid2D


@tf.function
def load_annotation(annotation_line: tf.Tensor, jing=False):
    '''
    read image and gt_boxes.
    return image, bboxes. box in [l, t, r, b]
    '''
    split_char = ' '
    if jing:
        split_char = '#'
    tf_lines = tf.strings.split(annotation_line, split_char)
    # tf_lines[1]: width, tf_lines[2]: height
    # left top  width height class
    true_box = tf.strings.split(tf_lines[3:], ',')  # (t, 5)
    true_box = tf.strings.to_number(true_box)

    # read and decode
    image_data = tf.io.read_file(tf_lines[0])
    image_data = tf.image.decode_image(image_data, expand_animations=False)
    # get original image shape
    img_shape = tf.shape(image_data)
    # grayscale_to_rgb
    if img_shape[2] == 1:
        image_data = tf.image.grayscale_to_rgb(image_data)
    # convert to float32 [0, 1.]
    image_data = tf.image.convert_image_dtype(image_data, dtype=tf.float32)

    gt_boxes = true_box.to_tensor()
    num_box = tf.shape(gt_boxes)[0]
    if num_box > 0:
        box_lt = gt_boxes[::, 0:2]
        box_rb = gt_boxes[::, 2:4]
        gt_boxes = tf.concat([box_lt, box_rb, gt_boxes[::, 4:5]], axis=-1)
    else:
        gt_boxes = tf.zeros([0, 5])
    return image_data, gt_boxes


# (
#     input_signature=[tf.TensorSpec(shape=(None, None, None), dtype=tf.float32), tf.TensorSpec(shape=(None, None, 2), dtype=tf.float32),
#                      tf.TensorSpec(shape=(2, ), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)]
# )
@tf.function
def location_to_box(model_out, lstGrid, input_shape, score_threshold):
    """
    model_out - list of [h, w, a]
    lstGrid   - list of [h, w, 2]
    return:
        pred_bboxes:   [n, 4]
        all_cls_score: [n, c]
    """
    if (len(model_out) == 2):
        stride = tf.constant([32, 16], dtype=tf.float32)
    else:
        stride = tf.constant([32, 16, 8], dtype=tf.float32)
    pred_bboxes = []
    for i in range(len(model_out)):
        grid = lstGrid[i]
        prediction = model_out[i]
        obj_conf = tf.math.sigmoid(prediction[..., 0:1])
        cls_conf = tf.math.sigmoid(prediction[..., 5:])
        # cls_conf = prediction[..., 5:]
        # print(prediction[..., 0])
        # object_score > .6. return (n, d)
        valid_index = tf.where(obj_conf[..., 0] > score_threshold)
        # print('levle: {}, valid_index:{}'.format(i, valid_index))
        # targets = tf.gather_nd(prediction[..., 1:5], valid_index)
        # print('levle: {}, target:{}'.format(i, targets))
        # target2box
        pred_xy = (prediction[..., 1:3] + grid) * stride[i] / input_shape
        pred_wh = tf.math.exp(tf.clip_by_value(prediction[..., 3:5], -4.5, 4.5)) * stride[i] / input_shape
        pred_lt = pred_xy - pred_wh / 2
        pred_rb = pred_xy + pred_wh / 2
        prediction = tf.concat([obj_conf, pred_lt, pred_rb, cls_conf], axis=-1)
        
        valid_boxes = tf.gather_nd(prediction, valid_index)  # (n, 1+4+num_classes)
        # print('levle: {}, valid_boxes:{}'.format(i, valid_boxes[..., 5:]))
        pred_bboxes.append(valid_boxes)
    pred_bboxes = tf.concat(pred_bboxes, axis=0)
    all_cls_score = pred_bboxes[..., 0:1] * pred_bboxes[..., 5:]  # 
    # valid left top right bottom
    box_lt = pred_bboxes[..., 1:3]
    box_rb = pred_bboxes[..., 3:5]
    box_lt = tf.where(box_lt < 0, .0, box_lt)
    box_rb = tf.where(box_rb > 1, 1.0, box_rb)
    pred_bboxes = tf.concat([box_lt, box_rb], axis=-1)
    return pred_bboxes, all_cls_score


@tf.function(
    input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32), tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                     tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32)]
)
def process_nms(pred_bboxes, all_cls_score, score_threshold, iou_threshold, num_classes=20):
    """ NMS """
    """
        pred_bboxes: [n, 4]
        all_cls_score: [n, c], c--num of classes
    """
    # y1 x1 y2 x2
    to_nms_boxes = tf.concat([pred_bboxes[..., 1:2], pred_bboxes[..., 0:1], pred_bboxes[..., 3:4], pred_bboxes[..., 2:3]], axis=-1)
    
    def loop_cond(i_cls, output, output_i):
        return tf.math.less(i_cls, num_classes)
    
    def loop_nms(i_cls, output, output_count):
        # mask = tf.where(cls_label == i, True, False)
        # per_cls_score = tf.boolean_mask(max_cls_sorce, mask)
        # if(len(per_cls_score)) == 0:
        #     continue
        # per_pred_bboxes = tf.boolean_mask(pred_bboxes, mask)
        # per_to_nms_boxes = tf.boolean_mask(to_nms_boxes, mask)
        selected = tf.image.non_max_suppression(to_nms_boxes, all_cls_score[..., i_cls], 40, iou_threshold, score_threshold)
        num_selected = tf.shape(selected)[0]
        selected = tf.expand_dims(selected, axis=-1)
        if num_selected > 0:
            boxes = tf.gather_nd(pred_bboxes, selected)
            # print('boxes:', boxes)
            # boxes = tf.expand_dims(boxes, axis=0)
            # image_data = tf.image.draw_bounding_boxes(image_data, boxes, frm_colors)
            selected_scores = tf.gather_nd(all_cls_score[..., i_cls], selected)
            selected_scores = tf.expand_dims(selected_scores, axis=-1)
            class_label = i_cls + tf.zeros([num_selected, 1], dtype=tf.int32)
            class_label = tf.cast(class_label, tf.float32)
            class_boxes = tf.concat([boxes, class_label, selected_scores], axis=-1)
            output = output.write(output_count, class_boxes)
            output_count += 1
        return i_cls + 1, output, output_count
    output_count = 0
    out_bboxes = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    _, out_bboxes, output_count = tf.while_loop(cond=loop_cond, body=loop_nms, loop_vars=[0, out_bboxes, output_count])
    if output_count > 0:
        out_bboxes = out_bboxes.concat()
    else:
        out_bboxes = tf.zeros((0, 6), dtype=tf.float32)
    return output_count, out_bboxes


def createmodel(input_shape, num_classes, weight_path, avgweight_path):
    # model_body
    model_body, blockoutputs = create_detectorRDD(input_shape, num_classes, True, deepsupervise=True, weight_decay=0.0005)
    model_body.load_weights(weight_path)
    checkpoint = tf.train.Checkpoint(averaged_weights=model_body.variables)  # model_body.trainable_weights variables
    # path_chpt = tf.train.latest_checkpoint('F:\\averaged')
    # path_chpt = 'F:\\averaged\\checkpoint-76'
    # print('path_chpt:', path_chpt)
    if len(avgweight_path) > 0:
        status = checkpoint.restore(avgweight_path)
        status.expect_partial()
        # print('status consumed:', status.assert_consumed())
        print('status matched:', status.assert_existing_objects_matched())
    return model_body


def createvalidateset(input_shape, tta, batch_size, annotation_path):
    with open(annotation_path) as f:
        annotation_lines = f.readlines()
    if annotation_lines[0][-1] == '\n':
        print('annotation_lines')
        for i in range(len(annotation_lines)):
            annotation_lines[i] = annotation_lines[i][:-1]

    def annotation_process(annotation_line):
        '''
        '''
        tf_lines = tf.strings.split(annotation_line)
        # load and resize images
        image, boxes = load_annotation(annotation_line)
        img_shape = tf.shape(image)[:2]
        
        # get padding offset
        scale = tf.reduce_min(input_shape / img_shape)
        scale = tf.cast(scale, dtype=tf.float32)
        aspect_ratio = tf.cast(img_shape, tf.float32) * scale
        aspect_ratio = tf.cast(aspect_ratio, tf.int32)
        image = tf.image.resize(image, aspect_ratio)  # method=ResizeMethod.BILINEAR
        # padding
        padding_lt = (input_shape - aspect_ratio) // 2
        padding_rb = input_shape - aspect_ratio - padding_lt
        image = tf.pad(image, [[padding_lt[0], padding_rb[0]], [padding_lt[1], padding_rb[1]], [0, 0]], constant_values=114/255.0)

        # correct box
        padding_lt = tf.cast(padding_lt, dtype=tf.float32)
        padding_lt = padding_lt[::-1]
        if tta:
            image_flip = tf.image.flip_left_right(image)
            return (image, image_flip, tf_lines[0], img_shape, padding_lt)
        else:
            return (image, tf_lines[0], img_shape, padding_lt)

    x_dataset = tf.data.Dataset.from_tensor_slices(annotation_lines)
    x_dataset = x_dataset.map(annotation_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    x_dataset = x_dataset.batch(batch_size, drop_remainder=False).repeat(1)
    return x_dataset, len(annotation_lines)


@tf.function(
    input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32)]
)
def FlipAugmentedConvert(pred_bboxes):
    """
    pred_bboxes:   [n, 4]
    all_cls_score: [n, c]
    """
    left = 1.0 - pred_bboxes[:, 2:3]
    right = 1.0 - pred_bboxes[:, 0:1]
    return tf.concat([left, pred_bboxes[:, 1:2], right, pred_bboxes[:, 3:4]], axis=-1)


def main(annotation_path, weight_path, avgweight_path, dst_path):
    batch_size = 32
    TTA = True       # Test Time Augmentation
    num_classes = 4

    # ddc2020 0.5387
    input_shape = (704, 704)
    tta_shape = (640, 640)
    score_threshold = .3
    iou_threshold = .99

    stride = [32, 16, 8]
    lstGrid = []
    for i in range(len(stride)):
        h = input_shape[0]//stride[i]
        w = input_shape[1]//stride[i]
        grid = generate_grid2D(h, w)
        lstGrid.append(tf.cast(grid, tf.float32))
    
    lstGrid_resize = []
    for i in range(len(stride)):
        h = tta_shape[0]//stride[i]
        w = tta_shape[1]//stride[i]
        grid = generate_grid2D(h, w)
        lstGrid_resize.append(tf.cast(grid, tf.float32))

    # load detector
    model_body = createmodel(input_shape, num_classes, weight_path, avgweight_path)

    # prepare data
    x_dataset, datacount = createvalidateset(input_shape, TTA, batch_size, annotation_path)
    # open result file
    outFile = open(dst_path, 'w', newline="")
    csv_writer = csv.writer(outFile)
    # start = time.time()
    step = (datacount / batch_size)
    if step > (datacount // batch_size):
        step += 1
    step = int(step)
    print('datacount:', datacount, 'step:', step)
    # for each batch
    iterator = iter(x_dataset)
    for i in range(step):
        print('progress: ', i, end='\r')
        testing_data = iterator.get_next()
        # (tf_lines[0], image_data, detal, scale)

        # for each image
        if i < step - 1:
            image_count = batch_size
        else:
            image_count = datacount - i * batch_size
            print('image_count:', image_count)
        
        bounding_boxes_batch, boxes_count_batch = [], []
        # inference
        if TTA is True:
            model_out = model_body.predict_on_batch(testing_data[0])
            model_out_flip = model_body.predict_on_batch(testing_data[1])
            aug_image_resize = tf.image.resize(testing_data[0], tta_shape)
            model_out_resize = model_body.predict_on_batch(aug_image_resize)
            for j in range(image_count):
                image_pred = [model_out[0][j], model_out[1][j]]  # , model_out[2][j]
                pred_bboxes, all_cls_score = location_to_box(image_pred, lstGrid, input_shape, score_threshold)
                # print('all_cls_score:', all_cls_score)
                # prediction: list of np.array (6). l, t, r, b, c, score.
                # 水平翻转后, 转换坐标
                image_pred_flip = [model_out_flip[0][j], model_out_flip[1][j]]  # , model_out_flip[2][j]
                pred_bboxes_flip, all_cls_score_flip = location_to_box(image_pred_flip, lstGrid, input_shape, score_threshold)
                pred_bboxes_flip = FlipAugmentedConvert(pred_bboxes_flip)
                # resize image
                image_pred_resize = [model_out_resize[0][j], model_out_resize[1][j]]  # , model_out_resize[2][j]
                pred_bboxes_resize, all_cls_score_resize = location_to_box(image_pred_resize, lstGrid_resize, tta_shape, score_threshold)

                pred_bboxes = tf.concat([pred_bboxes, pred_bboxes_flip, pred_bboxes_resize], axis=0)
                all_cls_score = tf.concat([all_cls_score, all_cls_score_flip, all_cls_score_resize], axis=0)
                output_count, out_bboxes = process_nms(pred_bboxes, all_cls_score, score_threshold, iou_threshold, num_classes)
                bounding_boxes_batch.append(out_bboxes)
                boxes_count_batch.append(output_count)
        else:
            model_out = model_body.predict_on_batch(testing_data[0])
            for j in range(image_count):
                image_pred = [model_out[0][j], model_out[1][j], model_out[2][j]]
                pred_bboxes, all_cls_score = location_to_box(image_pred, lstGrid, input_shape, score_threshold)
                # print('all_cls_score:', all_cls_score)
                # prediction: list of np.array (6). l, t, r, b, c, score.
                output_count, out_bboxes = process_nms(pred_bboxes, all_cls_score, score_threshold, iou_threshold, num_classes)
                bounding_boxes_batch.append(out_bboxes)
                boxes_count_batch.append(output_count)

        for j in range(len(bounding_boxes_batch)):
            # print(out_bboxes)
            # for N boxes
            if TTA:
                image_path = testing_data[2][j]   # image_path
            else:
                image_path = testing_data[1][j]   # image_path
            image_path = str(image_path.numpy())
            begin_pos = image_path.rindex('/')
            # print('begin_pos:', begin_pos)
            image_id = image_path[begin_pos+1:-1]
            predictions = ''
            if TTA:
                image_shape = testing_data[3][j]   # image_shape
                padding_lt = testing_data[4][j]   # padding
            else:
                image_shape = testing_data[2][j]   # image_shape
                padding_lt = testing_data[3][j]   # padding
            scale = tf.reduce_min(input_shape / image_shape)
            scale = tf.cast(scale, dtype=tf.float32)
            
            boxes = bounding_boxes_batch[j]
            # if boxes.shape[0] > 5:
            #     boxes = boxes[:5]
            if boxes_count_batch[j] > 0:
                bboxes_lt = boxes[:, :2] * input_shape - padding_lt
                bboxes_rb = boxes[:, 2:4] * input_shape - padding_lt
                bboxes_lt /= scale
                bboxes_rb /= scale
                bboxes_lt = tf.cast(bboxes_lt+.5, tf.int32)
                bboxes_rb = tf.cast(bboxes_rb+.5, tf.int32)
                bboxes_lt = tf.where(bboxes_lt < 0, 0, bboxes_lt)
                bboxes_r = tf.where(bboxes_rb[:, 0:1] > image_shape[1], image_shape[1], bboxes_rb[:, 0:1])
                bboxes_b = tf.where(bboxes_rb[:, 1:2] > image_shape[0], image_shape[0], bboxes_rb[:, 1:2])
                bboxes_rb = tf.concat([bboxes_r, bboxes_b], axis=-1)
                for m in range(tf.shape(boxes)[0]):
                    if len(predictions) > 0:
                        clsid = ' %d' % (boxes[m, 4] + 1)
                    else:
                        clsid = '%d' % (boxes[m, 4] + 1)
                    if int(clsid) > 4:
                        continue
                    coord = ' {0} {1} {2} {3}'.format(bboxes_lt[m, 0], bboxes_lt[m, 1], bboxes_rb[m, 0], bboxes_rb[m, 1])
                    # clsscore = ',%f' % boxes[m, 5]
                    predictions += clsid + coord
                    
            csv_writer.writerow([image_id, predictions])

    outFile.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation_path', type=str, help='annotation path')
    parser.add_argument('--weights', type=str, help='model weights')
    parser.add_argument('--avgweights', type=str, help='model averaged weights', default='')
    parser.add_argument('--dst', type=str, help='prediction path')

    args = parser.parse_args(sys.argv[1:])
    main(args.annotation_path, args.weights, args.avgweights, args.dst)
# python eval_grddc.py --annotation_path annotation/rdd_test1_20.txt --weights weights/ep180-loss5.536.h5 --dst res_grddc2.csv --avgweights weights/model-180
# https://github.com/HYK-baby/Road_Damage_Detection_with_YOLO.git