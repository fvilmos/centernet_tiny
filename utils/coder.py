'''************************************************************************** 
encoder decoder and IOU helpers
Author: fvilmos, https://github.com/fvilmos
***************************************************************************'''

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import cv2

def center2corner(bboxes):
    '''
    transform bbox representation from xc,yc,w,h (xcenter, ycenter, width, height) => x0,y0,x1,y1 (left-top, right-down coordinates)
    '''
    if not tf.is_tensor(bboxes):
       bboxes = tf.convert_to_tensor(bboxes, dtype=tf.float32)
        
    cx = bboxes[:, 0]
    cy = bboxes[:, 1]
    wc = bboxes[:, 2]
    hc = bboxes[:, 3]
    x1 = tf.subtract(cx , tf.multiply(0.5, wc))
    y1 = tf.subtract(cy , tf.multiply(0.5, hc))
    x2 = tf.add(cx , tf.multiply(0.5, wc))
    y2 = tf.add(cy , tf.multiply(0.5, hc))
    
    bboxes = tf.stack((x1, y1, x2, y2), axis=-1)
    return bboxes

def compute_iou(gt, pred):
    """
    returns IOU for to tesors, in bbox format of [xc,yc,w,h]
    """
    gt = tf.cast(gt, dtype=tf.float32)
    pred = tf.cast(pred, dtype=tf.float32)
    
    gt_corners = center2corner(gt)
    pred_corners = center2corner(pred)
    lu = tf.maximum(gt_corners[:, None, :2], pred_corners[:, :2])
    rd = tf.minimum(gt_corners[:, None, 2:], pred_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    gt_area = gt[:, 2] * gt[:, 3]
    pred_area = pred[:, 2] * pred[:, 3]
    union_area = tf.maximum(gt_area[:, None] + pred_area - intersection_area, 1e-8)

    # calculate GIOU
    lu = tf.minimum(gt_corners[:, None, :2], pred_corners[:, :2])
    rd = tf.maximum(gt_corners[:, None, 2:], pred_corners[:, 2:])
    
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)


def nms (heat_map, kernel=(3,3)):
    """
    Helper function, does the NMS on a heatmap
    """
    #idea: https://towardsdatascience.com/centernet-explained-a7386f368962#:~:text=CenterNet%20is%20an%20anchorless%20object%20detection%20architecture.,enables%20a%20much%20faster%20inference.
    hm_pool = tf.nn.max_pool2d(heat_map, kernel, strides=(1,1), padding="SAME")
    hm = tf.where(tf.equal(hm_pool,heat_map), heat_map, tf.zeros_like(heat_map))
    return hm * heat_map


def generate_heatmap(keypoints, radius=[10], img_size=(256,256,3), sigma=0, normalize=True, strides=4):
    """
    Generate heatmap with Gaussian distribution of the centerpoint location
    """
    hm = np.zeros(shape=img_size[:2], dtype=np.float32)
    radius[0] =radius[0] // strides

    for i,kp in enumerate(keypoints):
        
        # creat a gaussian kernel
        radius[i] = 1 if radius[i] <= 0 else radius[i]
        gkernel = cv2.getGaussianKernel(int(radius[i]), sigma=sigma)
        gkernel2d= np.dot(gkernel, gkernel.T)

        # apply over img, [y,x] on hm, [x,y] on kp
        reminder = radius[i] % 2
        h_radius = int(0.5 *  radius[i])
        x_begin, x_end = np.max([0,kp[0]]) - h_radius, np.min([hm.shape[1],kp[0]]) + h_radius + reminder
        y_begin, y_end =  np.max([0,kp[1]]) - h_radius, np.min([hm.shape[0],kp[1]]) + h_radius + reminder

        #resize gaussian kernel boundaries
        gkernel2d = gkernel2d[:,np.abs(x_begin):] if x_begin < 0 else gkernel2d
        gkernel2d = gkernel2d[:,:x_end- hm.shape[1]] if x_end > hm.shape[1] else gkernel2d
        gkernel2d = gkernel2d[np.abs(y_begin):,:] if y_begin < 0 else gkernel2d
        gkernel2d = gkernel2d[:y_end- hm.shape[0],:] if y_end > hm.shape[0] else gkernel2d

        # keep gaussian kernel in the heatmap dimensions
        vals = np.array([y_begin,y_end,x_begin,x_end])
        vals = np.clip(vals,0,hm.shape[0])

        hm[vals[0]:vals[1], vals[2]:vals[3]] = gkernel2d

        # normalize between [0,1]
        if normalize:
            hm = np.interp(hm,(hm.min(), hm.max()), (0,0.6))
            hm[kp[1], kp[0]] = 1.0
            
            
    return hm


def encoder_wrapper(input_shape=[64,64], strides=4, classes=1, sigma=0, normalize=False):
    """
    Encode input information to heatmas
    i.e. for [0.,250.,250.,20.,40.] input values generate the corresponding heatmaps with given class number + size + offset values, with , hight    
    """

    def encoder (annotations):
        # new shape
        y_shape = input_shape[0] // strides
        x_shape = input_shape[1] // strides

        
        heatmap = np.zeros(shape=[y_shape,x_shape, classes], dtype = np.float32)
        offset = np.zeros(shape=[y_shape,x_shape, 2], dtype = np.float32)
        sizes = np.zeros(shape=[y_shape,x_shape, 2], dtype = np.float32)

        for ann in annotations:

            # correction
            cor_x =  ann[1] % strides
            cor_y =  ann[2] % strides
            
            # resize
            ann = np.array(ann) / np.array([1,strides,strides,1,1])

            # c, x,y,w,h
            c, xc, yc, w, h = ann

            radius = np.min([w/2,h/2])

            hm = generate_heatmap([[int(xc),int(yc)]], [int(radius)], [y_shape, x_shape,1],sigma=sigma, normalize=True,strides=strides)
            heatmap[...,int(c)] = hm

            offset[int(yc),int(xc)] = [cor_x, cor_y]
      
            # create sizes [x,y]
            sizes[int(yc),int(xc)] = [w, h]
             

        # normalize values to the input shape
        if normalize:
            offset /= np.max([input_shape[0], input_shape[1]])
            sizes /= np.max([input_shape[0], input_shape[1]])

        return heatmap, offset, sizes

    return encoder

def decode_detections(heat_map, offset_map, size_map, strides, top_k=5):
    """
    takes the generated heatmaps from the model and decodes it to detections
    use the top_k number to filter the number of the best detections
    """
    
    # get heatmap shape
    batch, height, width, classes = heat_map.shape

    # compute heatmap nms
    heat_map = tf.cast(nms(heat_map), dtype=tf.float32)

    # b,h,w,c -> b,hxwxc
    hm_flat = tf.reshape(heat_map,[batch,-1])

    # b,h,w,2 -> b,hxw,2
    offset_flat = tf.reshape(offset_map,[batch,-1,2])

    # b,h,w,2 -> b,hxw,2
    size_flat = tf.reshape(size_map,[batch,-1,2])

    def process_sample(data_input):
        # get inputs
        hm_s, offs_s, size_s = data_input

        # get scores and indices
        # format h,w,c -> hxwxc
        top_k_scores, top_k_indices = tf.math.top_k(hm_s, k=top_k, sorted=True)

        # get classes
        top_k_c = tf.cast(top_k_indices % classes, tf.float32)
        
        # idexes follow the class format,ie. [a,b], [a,b] => a,a,b,b
        top_k_indices = tf.cast(top_k_indices // classes, tf.int32)

        # get xc,yc 
        top_k_xc = tf.cast(top_k_indices % width, tf.float32)
        top_k_yc = tf.cast(top_k_indices // width, tf.float32)

        # resize to input image dimensions
        top_k_xc = tf.multiply(top_k_xc, strides)
        top_k_yc = tf.multiply(top_k_yc, strides)
        
        # get offsets, compensate repojection
        top_k_offs = tf.gather(offs_s,top_k_indices)
        top_k_xc = tf.add(top_k_xc, top_k_offs[...,0])
        top_k_yc = tf.add(top_k_yc, top_k_offs[...,1])

        # get bbox sizes, [w,h]
        top_k_size = tf.gather(size_s,top_k_indices)

        # s, c, xc, yc, w, h
        return tf.stack([top_k_scores, top_k_c, top_k_xc, top_k_yc, top_k_size[...,0], top_k_size[...,1]], axis=-1)
        
    batch_samples = tf.map_fn(process_sample, [hm_flat, offset_flat, size_flat], dtype=tf.float32)[0]

    return batch_samples
