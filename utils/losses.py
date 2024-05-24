'''************************************************************************** 
losses
Author: fvilmos, https://github.com/fvilmos
***************************************************************************'''
from .coder import nms
import tensorflow as tf
import tensorflow.keras as keras

def wrapper_focal_loss_hm(alpha=2.0, gamma=4.0, low_threshold=0.1):
    @tf.function
    def focal_loss_hm(hm_true, hm_pred):  
        '''
        Focal Loss for Dense Object Detection: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/losses/CategoricalFocalCrossentropy
        fl(p_t) = alpha * (1-p_t)^gamma*log(p_t)
        cross entroty is log(p_t)
        p_t = if y_true==1, output else 1-output, same is for alpha
        '''
        
        alpha = 2.0
        gamma = 4.0
        pos_mask = tf.cast(tf.equal(hm_true, 1), tf.float32)
    
        neg_mask = tf.cast(tf.less(hm_true, 1), tf.float32)
        neg_mask = tf.cast(tf.greater_equal(neg_mask, low_threshold), tf.float32)
        
        neg_weights = tf.pow(1.0 - hm_true, gamma)
    
        pos_loss = -tf.math.log(tf.clip_by_value(hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(1 - hm_pred, alpha) * pos_mask
        neg_loss = -tf.math.log(tf.clip_by_value(1 - hm_pred, 1e-4, 1. - 1e-4)) * tf.pow(hm_pred, alpha) * neg_weights * neg_mask
    
        num_pos = tf.reduce_sum(pos_mask[...,:])
        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)
    
        cls_loss = tf.cond(tf.greater(num_pos, 0), lambda: (pos_loss + neg_loss) / num_pos, lambda: neg_loss)
        return cls_loss
    
    return focal_loss_hm

@tf.function
def reg_loss_by_label(y_gt, y_pred, heat_map, threshold=0.5):
    '''
    compute regression loss on a heatmp
    '''
    b,_,_,_ = y_gt.shape

    heat_map = tf.cast(nms(heat_map), dtype=tf.float32)
    
    mask = tf.expand_dims(tf.reduce_sum(heat_map,axis=-1), -1) 
    mask = tf.where(tf.greater_equal(mask,threshold),1.0, 0.0)
    
    mask = tf.reshape(mask,[b,-1,1])

    mask = tf.concat([mask, mask], axis=-1)
    index = tf.where(tf.greater_equal(mask, 1.0))

    y_gt = tf.reshape(y_gt,[b,-1,2])
    
    val_gt = tf.gather_nd(y_gt, index)
    val_gt = tf.reshape(val_gt,[1,-1,2])

    y_pred = tf.reshape(y_pred,[b,-1,2])
    val_pred = tf.gather_nd(y_pred, index)
    val_pred = tf.reshape(val_pred,[1,-1,2])

    num = tf.cast(tf.math.count_nonzero(mask), dtype=tf.float32)
    
    loss_x = keras.losses.mean_absolute_error(val_gt[...,0],val_pred[...,0])
    loss_y = keras.losses.mean_absolute_error(val_gt[...,1],val_pred[...,1])

    loss_x = loss_x / (num + 1e-15)
    loss_y = loss_y / (num + 1e-15)
    
    return  loss_x + loss_y
