import tensorflow as tf

def partitioned_non_max_suppression_padded(boxes,
                                           scores,
                                           max_output_size,
                                           iou_threshold=0.5,
                                           score_threshold=float('-inf')):

  num_boxes = tf.shape(boxes)[0]
  pad = tf.cast(
      tf.ceil(tf.cast(num_boxes, tf.float32) / _NMS_TILE_SIZE),
      tf.int32) * _NMS_TILE_SIZE - num_boxes

  scores, argsort_ids = tf.nn.top_k(scores, k=num_boxes, sorted=True)
  boxes = tf.gather(boxes, argsort_ids)
  num_boxes = tf.shape(boxes)[0]
  num_boxes += pad
  boxes = tf.pad(
      tf.cast(boxes, tf.float32), [[0, pad], [0, 0]], constant_values=-1)
  scores = tf.pad(tf.cast(scores, tf.float32), [[0, pad]])

  # mask boxes to -1 by score threshold
  scores_mask = tf.expand_dims(
      tf.cast(scores > score_threshold, boxes.dtype), axis=1)
  boxes = ((boxes + 1.) * scores_mask) - 1.

  boxes = tf.expand_dims(boxes, axis=0)
  scores = tf.expand_dims(scores, axis=0)

  def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
    return tf.logical_and(
        tf.reduce_min(output_size) < max_output_size,
        idx < num_boxes // _NMS_TILE_SIZE)

  selected_boxes, _, output_size, _ = tf.while_loop(
      _loop_cond, _suppression_loop_body,
      [boxes, iou_threshold,
       tf.zeros([1], tf.int32),
       tf.constant(0)])
  idx = num_boxes - tf.cast(
      tf.nn.top_k(
          tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
          tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
      tf.int32)
  idx = tf.minimum(idx, num_boxes - 1 - pad)
  idx = tf.reshape(idx + tf.reshape(tf.range(1) * num_boxes, [-1, 1]), [-1])
  num_valid_boxes = tf.reduce_sum(output_size)
  return (idx, num_valid_boxes, tf.reshape(boxes, [-1, 4]),
          tf.reshape(scores, [-1]), argsort_ids)