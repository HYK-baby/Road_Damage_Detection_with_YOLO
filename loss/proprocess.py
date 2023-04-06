import tensorflow as tf


@tf.function(
    input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=None, dtype=tf.float32)]
)
def get_iou(box1: tf.Tensor, box2: tf.Tensor):
    """
    """
    intersect_LT = tf.math.maximum(box1[..., :2], box2[..., :2])
    intersect_RB = tf.math.minimum(box1[..., 2:4], box2[..., 2:4])
    intersect_wh = tf.math.maximum(intersect_RB - intersect_LT, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    wh = tf.math.maximum(box1[..., 2:4] - box1[..., :2], 0.0)
    area1 = wh[..., 0] * wh[..., 1]
    wh = tf.math.maximum(box2[..., 2:4] - box2[..., :2], 0.0)
    area2 = wh[..., 0] * wh[..., 1]
    iou = intersect_area / (area1 + area2 - intersect_area + 1e-16)
    return iou


@tf.function(
    input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32),
                     tf.TensorSpec(shape=None, dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32)]
)
def get_diou(box1: tf.Tensor, box2: tf.Tensor, beta: tf.Tensor):
    """
    """
    center1 = (box1[..., 2:4] + box1[..., :2]) / 2
    center2 = (box2[..., 2:4] + box2[..., :2]) / 2
    d = center1-center2
    d = d[..., 0]**2 + d[..., 1]**2
    max_lt = tf.math.minimum(box1[..., :2], box2[..., :2])
    max_rb = tf.math.maximum(box1[..., 2:4], box2[..., 2:4])
    c = max_rb - max_lt
    c = c[..., 0]**2 + c[..., 1]**2
    # IOU
    intersect_LT = tf.math.maximum(box1[..., :2], box2[..., :2])
    intersect_RB = tf.math.minimum(box1[..., 2:4], box2[..., 2:4])
    intersect_wh = tf.math.maximum(intersect_RB - intersect_LT, 0.0)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

    wh = tf.math.maximum(box1[..., 2:4] - box1[..., :2], 0.0)
    area1 = wh[..., 0] * wh[..., 1]
    wh = tf.math.maximum(box2[..., 2:4] - box2[..., :2], 0.0)
    area2 = wh[..., 0] * wh[..., 1]
    iou = intersect_area / (area1 + area2 - intersect_area + 1e-16)

    return iou - (d/c)**beta


# @tf.function
def location_to_box(model_out, lstGrid, input_shape, score_threshold):
    """
    model_out - list of [b, h, w, a]
    return:
        pred_bboxes: [n, 4]
    """
    stride = tf.constant([32, 16, 8], dtype=tf.float32)
    pred_bboxes = []
    levels = len(model_out)
    for i in range(levels):
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


def process_prediction(pred_bboxes, all_cls_score, score_threshold, iou_threshold, num_classes=20):
    """ NMS """
    out_bboxes = []
    # cls_label = tf.argmax(all_cls_score, axis=-1)
    # max_cls_sorce = tf.reduce_max(all_cls_score, axis=-1)
    # y1 x1 y2 x2
    to_nms_boxes = tf.concat([pred_bboxes[..., 1:2], pred_bboxes[..., 0:1], pred_bboxes[..., 3:4], pred_bboxes[..., 2:3]], axis=-1)
    for i in range(num_classes):
        # mask = tf.where(cls_label == i, True, False)
        # per_cls_score = tf.boolean_mask(max_cls_sorce, mask)
        # if(len(per_cls_score)) == 0:
        #     continue
        # per_pred_bboxes = tf.boolean_mask(pred_bboxes, mask)
        # per_to_nms_boxes = tf.boolean_mask(to_nms_boxes, mask)
        selected = tf.image.non_max_suppression(to_nms_boxes, all_cls_score[..., i], 100, iou_threshold, score_threshold)
        num_selected = tf.shape(selected)[0]
        selected = tf.expand_dims(selected, axis=-1)
        if num_selected > 0:
            boxes = tf.gather_nd(pred_bboxes, selected)
            # print('boxes:', boxes)
            # boxes = tf.expand_dims(boxes, axis=0)
            # image_data = tf.image.draw_bounding_boxes(image_data, boxes, frm_colors)
            selected_scores = tf.gather_nd(all_cls_score[..., i], selected)
            selected_scores = tf.expand_dims(selected_scores, axis=-1)
            class_label = i + tf.zeros([num_selected, 1], dtype=tf.int32)
            class_label = tf.cast(class_label, tf.float32)
            class_boxes = tf.concat([boxes, class_label, selected_scores], axis=-1)
            out_bboxes.append(class_boxes)
    return out_bboxes


@tf.function(
    input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32), tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                     tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32)]
)
def soft_nms(pred_bboxes, all_cls_score, score_threshold, iou_threshold, num_classes):
    """
    pred_bboxes: [n, 4]
    all_cls_score: [n, num_classes]
    """
    output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    output_count = 0

    def loop_cond(i_cls, output, output_i):
        return tf.math.less(i_cls, num_classes)

    def loop_multilabel(i_cls, output, output_i):
        sorce_cls = all_cls_score[:, i_cls]
        mask = sorce_cls > score_threshold
        count = tf.where(mask, 1, 0)
        count = tf.reduce_sum(count)
        # tf.print('mask:', mask)
        # tf.print('count:', count)

        def loop_icond(i_box, sorce_cls, boxes_cls, output, output_i):
            return tf.math.less(i_box, count)

        def loop_nms(i_box, sorce_cls, boxes_cls, output, output_i):
            # sort
            ordering = tf.argsort(sorce_cls, direction='DESCENDING')
            # tf.print('ordering:', ordering)
            if sorce_cls[ordering[0]] > score_threshold:
                # output box
                boxes = boxes_cls[ordering[0]]
                selected_scores = sorce_cls[ordering[0]: ordering[0]+1]
                class_label = i_cls + tf.zeros((1, ), dtype=tf.int32)
                class_label = tf.cast(class_label, tf.float32)
                # print('boxes:', boxes)
                # print('class_label:', class_label)
                # print('selected_scores:', selected_scores)
                output_box = tf.concat([boxes, class_label, selected_scores], axis=-1)
                # tf.print('output_box:', output_box)
                output = output.write(output_i, output_box)
                output_i += 1

                # score decay when overlap too much.
                boxes_s = tf.gather(boxes_cls, ordering[1:])   # ==> shape=(n-1, 4)
                sorce_s = tf.gather(sorce_cls, ordering[1:])   # ==> shape=(n-1, 4)
                best_boxes = tf.expand_dims(boxes_cls[ordering[0]], 0)
                iou = get_iou(best_boxes, boxes_s)  # ==> shape=(1, n-1)
                # tf.print('best_boxes:', best_boxes)
                # tf.print('sorce_s:', sorce_s)
                # tf.print('iou:', iou)
                # linear weight
                weight = tf.where(iou > iou_threshold, 1 - iou, 1.0)
                # Gaussian weight
                # weight = tf.math.exp(0 - iou**2 / iou_threshold)
                sorce_cls = sorce_s * weight
                boxes_cls = boxes_s
            else:
                boxes_cls = tf.gather(boxes_cls, ordering[1:])   # ==> shape=(n-1, 4)
                sorce_cls = tf.gather(sorce_cls, ordering[1:])   # ==> shape=(n-1, 4)
            return i_box+1, sorce_cls, boxes_cls, output, output_i
        sorce_cls = sorce_cls[mask]
        boxes_cls = pred_bboxes[mask]
        # print('sorce_cls: ', sorce_cls)
        _, _, _, output, output_i = tf.while_loop(cond=loop_icond, body=loop_nms, loop_vars=[0, sorce_cls, boxes_cls, output, output_i])
        return i_cls + 1, output, output_i
    _, output, output_count = tf.while_loop(cond=loop_cond, body=loop_multilabel, loop_vars=[0, output, output_count])
    if output_count > 0:
        output = output.stack()
    else:
        output = tf.zeros((0, 6), dtype=tf.float32)
    # print('output: ', output)
    return output_count, output


@tf.function(
    input_signature=[tf.TensorSpec(shape=(None, 4), dtype=tf.float32), tf.TensorSpec(shape=(None, None), dtype=tf.float32),
                     tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.int32)]
)
def diou_nms(pred_bboxes, all_cls_score, score_threshold, iou_threshold, num_classes):
    """
    pred_bboxes: [n, 4]
    all_cls_score: [n, num_classes]
    """
    output = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
    output_count = 0

    def loop_cond(i_cls, output, output_i):
        return tf.math.less(i_cls, num_classes)

    def loop_multilabel(i_cls, output, output_i):
        sorce_cls = all_cls_score[:, i_cls]
        mask = sorce_cls > score_threshold
        count = tf.where(mask, 1, 0)
        count = tf.reduce_sum(count)
        # tf.print('mask:', mask)
        # tf.print('count:', count)

        def loop_icond(i_box, sorce_cls, boxes_cls, output, output_i):
            return tf.math.less(i_box, count)

        def loop_nms(i_box, sorce_cls, boxes_cls, output, output_i):
            # sort
            ordering = tf.argsort(sorce_cls, direction='DESCENDING')
            # tf.print('ordering:', ordering)
            if sorce_cls[ordering[0]] > score_threshold:
                # output box
                boxes = boxes_cls[ordering[0]]
                selected_scores = sorce_cls[ordering[0]: ordering[0]+1]
                class_label = i_cls + tf.zeros((1, ), dtype=tf.int32)
                class_label = tf.cast(class_label, tf.float32)
                # print('boxes:', boxes)
                # print('class_label:', class_label)
                # print('selected_scores:', selected_scores)
                output_box = tf.concat([boxes, class_label, selected_scores], axis=-1)
                # tf.print('output_box:', output_box)
                output = output.write(output_i, output_box)
                output_i += 1

                # score decay when overlap too much.
                boxes_s = tf.gather(boxes_cls, ordering[1:])   # ==> shape=(n-1, 4)
                sorce_s = tf.gather(sorce_cls, ordering[1:])   # ==> shape=(n-1, 4)
                best_boxes = tf.expand_dims(boxes_cls[ordering[0]], 0)
                diou = get_diou(best_boxes, boxes_s, .65)  # ==> shape=(1, n-1)
                # tf.print('best_boxes:', best_boxes)
                # tf.print('sorce_s:', sorce_s)
                # tf.print('iou:', iou)
                sorce_cls = tf.where(diou > iou_threshold, 0.0, sorce_s)
                boxes_cls = boxes_s
            else:
                if sorce_cls[ordering[0]] <= score_threshold:
                    i_box = 100000  # break
                else:
                    boxes_cls = tf.gather(boxes_cls, ordering[1:])   # ==> shape=(n-1, 4)
                    sorce_cls = tf.gather(sorce_cls, ordering[1:])   # ==> shape=(n-1, 4)
            return i_box+1, sorce_cls, boxes_cls, output, output_i
        sorce_cls = sorce_cls[mask]
        boxes_cls = pred_bboxes[mask]
        # print('sorce_cls: ', sorce_cls)
        _, _, _, output, output_i = tf.while_loop(cond=loop_icond, body=loop_nms, loop_vars=[0, sorce_cls, boxes_cls, output, output_i])
        return i_cls + 1, output, output_i
    _, output, output_count = tf.while_loop(cond=loop_cond, body=loop_multilabel, loop_vars=[0, output, output_count])
    if output_count > 0:
        output = output.stack()
    else:
        output = tf.zeros((0, 6), dtype=tf.float32)
    # print('output: ', output)
    return output_count, output
