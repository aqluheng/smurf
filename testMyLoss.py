import gin
import tensorflow as tf
from tensorflow_addons import image as tfa_image
import functools

def resampler_flat_gather(data, warp, name='flat_resampler'):
  """Resampler that avoids gather_nd which can be expensive on TPU.

  Computing gradients of gather_nd requires calling scatter_nd
  which is very slow on TPU and causes a large memory blowup.
  Empirically, this resampler produces a much lower memory footprint
  and faster inference time on the TPU by avoding gather_nd and instead
  using a flat gather. See tfa.image.resampler for more documentation.

  Args:
    data: float tf Tensor of shape b H W c, The source to differentiably
      resample from.
    warp: float tf Tensor of shape b h w 2, The set of coordinates to sample
      from data.
    name: str scope to put operations under.
  Returns:
    resampled_data: float tf Tensor of shape b h w c, The result of sampling
      data with warp.
  """
  with tf.name_scope(name):
    b, data_h, data_w, c = tf.unstack(tf.shape(data))
    _, warp_h, warp_w, _ = tf.unstack(tf.shape(warp))
    warp_x, warp_y = tf.unstack(warp, axis=-1)

    warp_shape = tf.shape(warp_x)
    warp_batch = tf.range(warp_shape[0], dtype=tf.int32)
    warp_batch = tf.reshape(warp_batch, (warp_shape[0], 1, 1))
    warp_batch = tf.broadcast_to(warp_batch, (b, warp_h, warp_w))
    warp_batch = tf.reshape(warp_batch, [-1])
    warp_x = tf.reshape(warp_x, [-1])
    warp_y = tf.reshape(warp_y, [-1])
    warp_floor_x = tf.math.floor(warp_x)
    warp_floor_y = tf.math.floor(warp_y)

    right_warp_weight = warp_x - warp_floor_x
    down_warp_weight = warp_y - warp_floor_y
    left_warp_weight = tf.subtract(
        tf.convert_to_tensor(1.0, right_warp_weight.dtype), right_warp_weight)
    up_warp_weight = tf.subtract(
        tf.convert_to_tensor(1.0, down_warp_weight.dtype), down_warp_weight)

    warp_floor_x = tf.cast(warp_floor_x, tf.int32)
    warp_floor_y = tf.cast(warp_floor_y, tf.int32)
    warp_ceil_x = tf.cast(tf.math.ceil(warp_x), tf.int32)
    warp_ceil_y = tf.cast(tf.math.ceil(warp_y), tf.int32)

    left_warp_weight = tf.expand_dims(left_warp_weight, -1)
    right_warp_weight = tf.expand_dims(right_warp_weight, -1)
    up_warp_weight = tf.expand_dims(up_warp_weight, -1)
    down_warp_weight = tf.expand_dims(down_warp_weight, -1)

    def flatten_warp(warp_y, warp_x):
      """Converts the warps from a 2D index to a 1D index."""
      output = tf.reshape(
          warp_batch * data_w * data_h + warp_y * data_w + warp_x, [-1])
      # Get a mask of the coordinates which go out of bounds.
      mask_y = tf.cast(
          tf.logical_and(warp_y >= 0, warp_y <= data_h - 1), dtype=data.dtype)
      mask_x = tf.cast(
          tf.logical_and(warp_x >= 0, warp_x <= data_w - 1), dtype=data.dtype)
      output = tf.clip_by_value(output, 0, b * data_h * data_w - 1)
      return output, tf.expand_dims(mask_y * mask_x, -1)

    up_left_warp, mask_up_left = flatten_warp(warp_floor_y, warp_floor_x)
    up_right_warp, mask_up_right = flatten_warp(warp_floor_y, warp_ceil_x)
    down_left_warp, mask_down_left = flatten_warp(warp_ceil_y, warp_floor_x)
    down_right_warp, mask_down_right = flatten_warp(warp_ceil_y, warp_ceil_x)
    flat_data = tf.reshape(data, (-1, c))

    up_left = tf.gather(flat_data, up_left_warp, axis=0) * mask_up_left
    up_right = tf.gather(flat_data, up_right_warp, axis=0) * mask_up_right
    down_left = tf.gather(flat_data, down_left_warp, axis=0) * mask_down_left
    down_right = tf.gather(flat_data, down_right_warp, axis=0) * mask_down_right
    result = (up_left * left_warp_weight + up_right * right_warp_weight
             ) * up_warp_weight + (down_left * left_warp_weight + down_right *
                                   right_warp_weight) * down_warp_weight
    return tf.reshape(result, (b, warp_h, warp_w, c))


def resampler(source, coords):
  # return tfa_image.resampler(source, coords)
  return resampler_flat_gather(source, coords)


def flow_to_warp(flow):
  """Compute the warp from the flow field.

  Args:
    flow: tf.tensor representing optical flow.

  Returns:
    The warp, i.e. the endpoints of the estimated flow.
  """

  # Construct a grid of the image coordinates.
  height, width = flow.shape.as_list()[-3:-1]
  i_grid, j_grid = tf.meshgrid(
      tf.linspace(0.0, height - 1.0, int(height)),
      tf.linspace(0.0, width - 1.0, int(width)),
      indexing='ij')
  grid = tf.stack([i_grid, j_grid], axis=2)

  # Potentially add batch dimension to match the shape of flow.
  if len(flow.shape) == 4:
    grid = grid[None]

  # Add the flow field to the image grid.
  if flow.dtype != grid.dtype:
    grid = tf.cast(grid, flow.dtype)
  warp = grid + flow
  return warp


def mask_invalid(coords, pad_h=0, pad_w=0):
  """Mask coordinates outside of the image.

  Valid = 1, invalid = 0.

  Args:
    coords: a 4D float tensor of image coordinates.
    pad_h: int, the amount of padding applied to the top of the image
    pad_w: int, the amount of padding applied to the left of the image

  Returns:
    The mask showing which coordinates are valid.
  """
  pad_h = float(pad_h)
  pad_w = float(pad_w)
  coords_rank = len(coords.shape)
  if coords_rank != 4:
    raise NotImplementedError()
  max_height = float(coords.shape[-3] - 1)
  max_width = float(coords.shape[-2] - 1)
  mask = tf.logical_and(
      tf.logical_and(coords[:, :, :, 0] >= pad_h,
                     coords[:, :, :, 0] <= max_height),
      tf.logical_and(coords[:, :, :, 1] >= pad_w,
                     coords[:, :, :, 1] <= max_width))
  mask = tf.cast(mask, dtype=tf.float32)[:, :, :, None]
  return mask


def resample(source, coords):
  """Resample the source image at the passed coordinates.

  Args:
    source: tf.tensor, batch of images to be resampled.
    coords: tf.tensor, batch of coordinates in the image. Coordinates should
      be between 0 and size - 1. Coordinates outside of this range are handled
      by interpolating with a background image filled with zeros in the same
      way that SAME size convolution works.

  Returns:
    The resampled image.
  """

  # Wrap this function because it uses a different order of height/width dims.
  orig_source_dtype = source.dtype
  if source.dtype != tf.float32:
    source = tf.cast(source, tf.float32)
  if coords.dtype != tf.float32:
    coords = tf.cast(coords, tf.float32)
  coords_rank = len(coords.shape)
  if coords_rank == 4:
    output = resampler(source, coords[:, :, :, ::-1])
    if orig_source_dtype != source.dtype:
      return tf.cast(output, orig_source_dtype)
    return output
  else:
    raise NotImplementedError()


def unsupervised_loss(images,
                      flows,
                      weights,
                      occlusion_estimation_fn,
                      only_forward = False,
                      selfsup_transform_fn = None,
                      fb_sigma_teacher = 0.003,
                      fb_sigma_student = 0.03,
                      smoothness_edge_weighting = 'gaussian',
                      smoothness_edge_constant = 150.0,
                      stop_gradient_mask = True,
                      selfsup_mask = 'gaussian',
                      smoothness_at_level = 2,
                      full_size_images = None,
                      crop_h = None,
                      crop_w = None,
                      pad_h = None,
                      pad_w = None):
  """Computes unsupervised SMURF losses.

  Args:
    images: (Unaugmented) images for which the flow fields are passed of shape
      [batch, time, height, width, channels].
    flows: Dictionary of flow fields. The key is given by (i, j, t), where i =
      reference image index, j = second image index, t = augementation/model
      type (e.g. augmented-student).
    weights: Dictionary holding the weights for the different loss functions. If
      a weight is not in the dictionary the loss will not be computed.
    occlusion_estimation_fn: Function to compute occlusions masks.
    only_forward: Flag indicating if only losses for the forward flow estimation
      should be computed.
    selfsup_transform_fn: List of self-supervion transform functions.
    fb_sigma_teacher: Sigma used for the gaussian self-supervision masking.
    fb_sigma_student: Sigma used for the gaussian self-supervision masking.
    smoothness_edge_weighting: Defines which function should be used for the
      edge-aware smoothing, can be either gaussian or exponential.
    smoothness_edge_constant: Constant used within the edge weighting function.
    stop_gradient_mask: Flag indicating if gradients should be stopped for the
      occlusion masks.
    selfsup_mask: Indicates what type of masking to use for the self-supervision
      can be either gaussian or ddflow.
    smoothness_at_level: Resolution level at which the smoothness loss should be
      applied.
    full_size_images: Optional uncropped images to use for warping. If uncropped
      images, crop_h, and crop_w are provided, we will use the full scale images
      to perform a warp. This has the benefit of allowing a loss to be computed
      for many flow vectors which move off the edge of the image.
    crop_h: Optional upper left row of the bounding box used to crop the images
      from the full_size_images.
    crop_w: Optional upper left col of the bounding box used to crop the images
      from the full_size_images.
    pad_h: Optional upper padding applied to the full_size_images. The padding
      was applied after the image was cropped and is not reflected in crop_h.
    pad_w: Optional left padding applied to the full_size_images. The padding
      was applied after the image was cropped and is not reflected in crop_w.

  Returns:
    Dictionary holding the calculated losses.
  """
  # Initialize unsupervised losses with zero.
  losses = {}
  for key in weights:
    losses[key] = tf.constant(0.)

  compute_loss_for_these_flows = ['augmented-student']
  # Count number of non self-sup pairs, for which we will apply the losses.
  num_pairs = sum( 
      [1.0 for (i, j, c) in flows if c in compute_loss_for_these_flows]) 

  # Ensure that smoothness_at_level is feasible, i.e. set smoothness_at_level
  # to be as close as possible to the chosen parameter. This means the current
  # default value of 2 will be modifed to 0 for raft with convex upsampling.
  smoothness_at_level = min(smoothness_at_level, 
                            len(flows[(0, 1, 'augmented-student')]) - 1)
  # num_pairs = 2.0
  # smoothness_at_level = 0
  
  # Always self supervise with the full resolution flows.
  selfsup_at_level = 0

  # Iterate over image pairs.
  for key in flows:
    time_i, time_j, c = key
    key_rev = (time_j, time_i, c)
    if (c not in compute_loss_for_these_flows or
        (only_forward and time_i > time_j)):
      continue

    if full_size_images is not None:
      flow = flows[key][0]
      height = flow.shape[-3]
      width = flow.shape[-2]
      full_height = full_size_images.shape[-3]
      full_width = full_size_images.shape[-2]
      batch_size = flow.shape[0]
      # TODO(smurf): Make work for batch size > 1
      with tf.control_dependencies([
          tf.compat.v1.assert_equal(batch_size, 1),
          tf.compat.v1.assert_greater_equal(full_height, height),
          tf.compat.v1.assert_greater_equal(full_width, width),
      ]):
        flow = tf.pad(flow, [[
            0, 0
        ], [crop_h[0] + pad_h[0], full_height - height - crop_h[0] - pad_h[0]
           ], [crop_w[0] + pad_w[0], full_width - width - crop_w[0] - pad_w[0]],
                             [0, 0]])
        flow.set_shape((batch_size, full_height, full_width, 2))
      warp = flow_to_warp(flow)
      valid_warp_mask = mask_invalid(warp, pad_h, pad_w)
      warped_image = resample(
          tf.stop_gradient(full_size_images[:, time_j]), warp)
      warped_image = tf.image.crop_to_bounding_box(warped_image,
                                                   pad_h[0] + crop_h[0],
                                                   pad_w[0] + crop_w[0], height,
                                                   width)
      valid_warp_mask = tf.image.crop_to_bounding_box(valid_warp_mask,
                                                      pad_h[0] + crop_h[0],
                                                      pad_w[0] + crop_w[0],
                                                      height, width)
    else:
      warp = flow_to_warp(flows[key][0])
      valid_warp_mask = mask_invalid(warp)
      warped_image = resample(tf.stop_gradient(images[:, time_j]), warp)

    occlusion_mask = occlusion_estimation_fn(
        forward_flow=flows[key][0], backward_flow=flows[key_rev][0])

    if stop_gradient_mask:
      mask_level0 = tf.stop_gradient(occlusion_mask * valid_warp_mask)
    else:
      mask_level0 = occlusion_mask * valid_warp_mask

    if 'census' in weights:
      # Loss based on the census transform.
      cen_loss = census_loss(
          image_a_bhw3=images[:, time_i],
          image_b_bhw3=warped_image,
          mask_bhw3=mask_level0)
      losses['census'] += weights['census'] * cen_loss / num_pairs

    # Compute smoothness losses.
    if 'smooth2' in weights or 'smooth1' in weights:
      # Configure function for the edge-aware weighting.
      def edge_weighting_fn(x):
        if smoothness_edge_weighting == 'gaussian':
          return tf.exp(-tf.reduce_mean(
              input_tensor=((smoothness_edge_constant * x)**2),
              axis=-1,
              keepdims=True))
        elif smoothness_edge_weighting == 'exponential':
          return tf.exp(-tf.reduce_mean(
              input_tensor=(abs(smoothness_edge_constant * x)),
              axis=-1,
              keepdims=True))
        else:
          raise ValueError('Only gaussian or exponential edge weighting '
                           'implemented.')

      # Resize multiple times for a smoother result.
      images_at_smoothness_level = images[:, time_i]
      for _ in range(smoothness_at_level):
        height = tf.shape(images_at_smoothness_level)[-3]
        width = tf.shape(images_at_smoothness_level)[-2]
        images_at_smoothness_level = resize(
            images_at_smoothness_level, (height) // 2, (width) // 2,
            is_flow=False)

      if 'smooth1' in weights:
        # Compute first-order smoohtness term loss.
        smooth_loss_1st = first_order_smoothness_loss(
            image=images_at_smoothness_level,
            flow=flows[key][smoothness_at_level],
            edge_weighting_fn=edge_weighting_fn)
        losses['smooth1'] += weights['smooth1'] * smooth_loss_1st / num_pairs

      if 'smooth2' in weights:
        # Compute second-order smoohtness term loss.
        smooth_loss_2nd = second_order_smoothness_loss(
            image=images_at_smoothness_level,
            flow=flows[key][smoothness_at_level],
            edge_weighting_fn=edge_weighting_fn)
        losses['smooth2'] += weights['smooth2'] * smooth_loss_2nd / num_pairs

    # Compute self-supervision loss.
    if 'selfsup' in weights:
      teacher_key = (time_i, time_j, 'original-teacher')
      student_key = (time_i, time_j, 'transformed-student')
      teacher_key_rev = (time_j, time_i, 'original-teacher')
      student_key_rev = (time_j, time_i, 'transformed-student')
      selfsup_loss = self_supervision_loss(
          teacher_flow=flows[teacher_key][selfsup_at_level],
          student_flow=flows[student_key][selfsup_at_level],
          teacher_backward_flow=flows[teacher_key_rev][selfsup_at_level],
          student_backward_flow=flows[student_key_rev][selfsup_at_level],
          selfsup_mask=selfsup_mask,
          selfsup_transform_fn=selfsup_transform_fn,
          fb_sigma_student=fb_sigma_student,
          fb_sigma_teacher=fb_sigma_teacher)
      losses['selfsup'] += weights['selfsup'] * selfsup_loss / num_pairs

  return losses

@tf.function
def resize(img, height, width, is_flow, mask=None):
  """Resize an image or flow field to a new resolution.

  In case a mask (per pixel {0,1} flag) is passed a weighted resizing is
  performed to account for missing flow entries in the sparse flow field. The
  weighting is based on the resized mask, which determines the 'amount of valid
  flow vectors' that contributed to each individual resized flow vector. Hence,
  multiplying by the reciprocal cancels out the effect of considering non valid
  flow vectors.

  Args:
    img: tf.tensor, image or flow field to be resized of shape [b, h, w, c]
    height: int, heigh of new resolution
    width: int, width of new resolution
    is_flow: bool, flag for scaling flow accordingly
    mask: tf.tensor, mask (optional) per pixel {0,1} flag

  Returns:
    Resized and potentially scaled image or flow field (and mask).
  """
  def _resize(image, mask=None):
    # _, orig_height, orig_width, _ = img.shape.as_list()
    orig_height = tf.shape(input=image)[1]
    orig_width = tf.shape(input=image)[2]

    if mask is not None:
      # multiply with mask, to ensure non-valid locations are zero
      image = tf.math.multiply(image, mask)
      # resize image
      img_resized = tf.compat.v2.image.resize(
          image, (int(height), int(width)), antialias=True)
      # resize mask (will serve as normalization weights)
      mask_resized = tf.compat.v2.image.resize(
          mask, (int(height), int(width)), antialias=True)
      # normalize sparse flow field and mask
      img_resized = tf.math.multiply(
          img_resized, tf.math.reciprocal_no_nan(mask_resized))
      mask_resized = tf.math.multiply(
          mask_resized, tf.math.reciprocal_no_nan(mask_resized))
    else:
      # normal resize without anti-alaising
      img_resized = tf.compat.v2.image.resize(image, (tf.cast(height,
                                                              tf.int32),
                                                      tf.cast(width,
                                                              tf.int32)))

    if is_flow:
      # If image is a flow image, scale flow values to be consistent with the
      # new image size.
      scaling = tf.reshape([
          float(height) / tf.cast(orig_height, tf.float32),
          float(width) / tf.cast(orig_width, tf.float32)
      ], [1, 1, 1, 2])
      img_resized *= scaling

    if mask is not None:
      return img_resized, mask_resized
    return img_resized

  # Apply resizing at the right shape.
  shape = img.shape.as_list()
  if img.shape.rank == 3:
    if mask is not None:
      img_resized, mask_resized = _resize(img[None], mask[None])
      return img_resized[0], mask_resized[0]
    else:
      return _resize(img[None])[0]
  if img.shape.rank == 4:
    # Input at the right shape.
    return _resize(img, mask)
  if img.shape.rank > 4:
    # Reshape input to [b, h, w, c], resize and reshape back.
    outer_shape = tf.shape(input=img)[:-3]
    required_shape = tf.concat([[-1], tf.shape(input=img)[-3:]], axis=0)
    img_flattened = tf.reshape(img, required_shape)
    if mask is not None:
      mask_flattened = tf.reshape(mask, required_shape)
      img_resized, mask_resized = _resize(img_flattened, mask_flattened)
    else:
      img_resized = _resize(img_flattened)
    final_shape = tf.concat(
        [outer_shape, tf.shape(input=img_resized)[-3:]], axis=0)
    result_img = tf.reshape(img_resized, final_shape)
    if mask is not None:
      final_mask_shape = tf.concat(
          [outer_shape, tf.shape(input=mask_resized)[-3:]], axis=0)
      result_mask = tf.reshape(mask_resized, final_mask_shape)
      return result_img, result_mask
    return result_img
  else:
    raise ValueError('Cannot resize an image of shape', shape)


def compute_range_map(flow,
                      downsampling_factor=1,
                      reduce_downsampling_bias=True,
                      resize_output=True):
  """Count how often each coordinate is sampled.

  Counts are assigned to the integer coordinates around the sampled coordinates
  using weights from bilinear interpolation.

  Args:
    flow: A float tensor of shape (batch size x height x width x 2) that
      represents a dense flow field.
    downsampling_factor: An integer, by which factor to downsample the output
      resolution relative to the input resolution. Downsampling increases the
      bin size but decreases the resolution of the output. The output is
      normalized such that zero flow input will produce a constant ones output.
    reduce_downsampling_bias: A boolean, whether to reduce the downsampling bias
      near the image boundaries by padding the flow field.
    resize_output: A boolean, whether to resize the output ot the input
      resolution.

  Returns:
    A float tensor of shape [batch_size, height, width, 1] that denotes how
    often each pixel is sampled.
  """

  # Get input shape.
  input_shape = flow.shape.as_list()
  if len(input_shape) != 4:
    raise NotImplementedError()
  batch_size, input_height, input_width, _ = input_shape

  flow_height = input_height
  flow_width = input_width

  # Apply downsampling (and move the coordinate frame appropriately).
  output_height = input_height // downsampling_factor
  output_width = input_width // downsampling_factor
  if downsampling_factor > 1:
    # Reduce the bias that comes from downsampling, where pixels at the edge
    # will get lower counts that pixels in the middle of the image, by padding
    # the flow field.
    if reduce_downsampling_bias:
      p = downsampling_factor // 2
      flow_height += 2 * p
      flow_width += 2 * p
      # Apply padding in multiple steps to padd with the values on the edge.
      for _ in range(p):
        flow = tf.pad(
            tensor=flow,
            paddings=[[0, 0], [1, 1], [1, 1], [0, 0]],
            mode='SYMMETRIC')
      coords = flow_to_warp(flow) - p
    # Update the coordinate frame to the downsampled one.
    coords = (coords + (1 - downsampling_factor) * 0.5) / downsampling_factor
  elif downsampling_factor == 1:
    coords = flow_to_warp(flow)
  else:
    raise ValueError('downsampling_factor must be an integer >= 1.')

  # Split coordinates into an integer part and a float offset for interpolation.
  coords_floor = tf.floor(coords)
  coords_offset = coords - coords_floor
  coords_floor = tf.cast(coords_floor, 'int32')

  # Define a batch offset for flattened indexes into all pixels.
  batch_range = tf.reshape(tf.range(batch_size), [batch_size, 1, 1])
  idx_batch_offset = tf.tile(
      batch_range, [1, flow_height, flow_width]) * output_height * output_width

  # Flatten everything.
  coords_floor_flattened = tf.reshape(coords_floor, [-1, 2])
  coords_offset_flattened = tf.reshape(coords_offset, [-1, 2])
  idx_batch_offset_flattened = tf.reshape(idx_batch_offset, [-1])

  # Initialize results.
  idxs_list = []
  weights_list = []

  # Loop over differences di and dj to the four neighboring pixels.
  for di in range(2):
    for dj in range(2):

      # Compute the neighboring pixel coordinates.
      idxs_i = coords_floor_flattened[:, 0] + di
      idxs_j = coords_floor_flattened[:, 1] + dj
      # Compute the flat index into all pixels.
      idxs = idx_batch_offset_flattened + idxs_i * output_width + idxs_j

      # Only count valid pixels.
      mask = tf.reshape(
          tf.compat.v1.where(
              tf.logical_and(
                  tf.logical_and(idxs_i >= 0, idxs_i < output_height),
                  tf.logical_and(idxs_j >= 0, idxs_j < output_width))), [-1])
      valid_idxs = tf.gather(idxs, mask)
      valid_offsets = tf.gather(coords_offset_flattened, mask)

      # Compute weights according to bilinear interpolation.
      weights_i = (1. - di) - (-1)**di * valid_offsets[:, 0]
      weights_j = (1. - dj) - (-1)**dj * valid_offsets[:, 1]
      weights = weights_i * weights_j

      # Append indices and weights to the corresponding list.
      idxs_list.append(valid_idxs)
      weights_list.append(weights)

  # Concatenate everything.
  idxs = tf.concat(idxs_list, axis=0)
  weights = tf.concat(weights_list, axis=0)

  # Sum up weights for each pixel and reshape the result.
  counts = tf.math.unsorted_segment_sum(
      weights, idxs, batch_size * output_height * output_width)
  count_image = tf.reshape(counts, [batch_size, output_height, output_width, 1])

  if downsampling_factor > 1:
    # Normalize the count image so that downsampling does not affect the counts.
    count_image /= downsampling_factor**2
    if resize_output:
      count_image = resize(
          count_image, input_height, input_width, is_flow=False)

  return count_image

def compute_occlusions_wang(backward_flow, downsampling_factor,
                            threshold):
  """Compute occlusion mask based on a rangemap.

  Args:
    backward_flow: Backward flow field of shape [batch, height, width, 2].
    downsampling_factor: Downsampling factor used for the range map computation.
    threshold: Indicates if thresholding should be used

  Returns:
    Occlusion mask of shape [batch, height, width, 1], where 1 are occluded
    locations and 0 are non-occluded.
  """
  range_map = compute_range_map(
      backward_flow,
      downsampling_factor=downsampling_factor,
      reduce_downsampling_bias=False,
      resize_output=False)
  if threshold:
    return 1.0 - tf.cast(range_map < 0.75, tf.float32)
  else:
    return 1.0 - tf.clip_by_value(range_map, 0.0, 1.0)

@tf.function
def compute_occlusions(forward_flow,
                       backward_flow,
                       occlusion_estimation = None,
                       occlusions_are_zeros = True,
                       occ_active = None,
                       boundaries_occluded = True):
  """Compute occlusion masks.

  Args:
    forward_flow: Forward flow field of shape [batch, height, width, 2].
    backward_flow: Backward flow field of shape [batch, height, width, 2].
    occlusion_estimation: Type of occlusion estimation that should be used.
    occlusions_are_zeros: Indicates if occlusions are indicated via 0 or 1.
    occ_active: Bool for each possible occlusion estimation type, indicating if
      occlusion estimation is active already or not.
    boundaries_occluded: If True, treat flow vectors pointing off the boundaries
      as occluded. Otherwise explicitly mark them as unoccluded.

  Returns:
    Occlusion mask of shape [batch, height, width, 1].
  """

  # Corresponding forward and backward flow.
  flow_ij = forward_flow
  flow_ji = backward_flow

  occlusion_mask = tf.zeros_like(flow_ij[Ellipsis, :1], dtype=tf.float32)

  occlusion_mask = compute_occlusions_wang(
        flow_ji, downsampling_factor=1, threshold=False)

  if not boundaries_occluded:
    warp = flow_to_warp(flow_ij)
    occlusion_mask = tf.minimum(occlusion_mask, mask_invalid(warp))

  return 1. - occlusion_mask if occlusions_are_zeros else occlusion_mask

@gin.configurable
def unsupervised_sequence_loss(
    images,
    flows_sequence,
    unsupervised_loss_fn,  # pylint:disable=g-bare-generic
    loss_decay = .8,
    supervision_weight = 0.05,
    mode = 'unsup_per_update',
    full_size_images = None,
    crop_h = None,
    crop_w = None,
    pad_h = None,
    pad_w = None):
  """Computes a unsupervised sequence loss."""
  loss_dict = {}

  def add_loss_dicts(old_dict, new_dict, decay):
    """Adds all losses in the dict considering a decay factor."""
    for key, value in new_dict.items():
      if key not in old_dict:
        old_dict[key] = value
      else:
        old_dict[key] = value + old_dict[key] * decay

  if mode == 'unsup_per_update':
    # Applies the same unsupervised loss for each update iteration.
    for flows in flows_sequence:
      # Compute the losses.
      loss_dict_one_flow = unsupervised_loss_fn(
          images=images,
          flows=flows,
          full_size_images=full_size_images,
          crop_h=crop_h,
          crop_w=crop_w,
          pad_h=pad_h,
          pad_w=pad_w)
      add_loss_dicts(loss_dict, loss_dict_one_flow, loss_decay)
  return loss_dict

from functools import partial
def build_selfsup_transformations(num_flow_levels=3,
                                  crop_height=0,
                                  crop_width=0,
                                  resize=True):
  """Apply augmentations to a list of student images."""
  def transform(images, is_flow, crop_height, crop_width, resize):

    height = images.shape[-3]
    width = images.shape[-2]

    op5 = tf.compat.v1.assert_greater(
        height,
        2 * crop_height,
        message='Image height is too small for cropping.')
    op6 = tf.compat.v1.assert_greater(
        width, 2 * crop_width, message='Image width is too small for cropping.')
    with tf.control_dependencies([op5, op6]):
      images = images[:, crop_height:height - crop_height,
                      crop_width:width - crop_width, :]
    if resize:
      images = resize(images, height, width, is_flow=is_flow)
      images.set_shape((images.shape[0], height, width, images.shape[3]))
    else:
      images.set_shape((images.shape[0], height - 2 * crop_height,
                        width - 2 * crop_width, images.shape[3]))
    return images

  max_divisor = 2**(num_flow_levels - 1)
  assert crop_height % max_divisor == 0
  assert crop_width % max_divisor == 0
  # Compute random shifts for different images in a sequence.
  return partial(
      transform,
      crop_height=crop_height,
      crop_width=crop_width,
      resize=resize)



# flow = [flow[0], flow[1], flow[2], flow[3], flow[4], flow[5]]
# flow[0] = dict_keys()
# flow[0].keys() = [(0, 1, 'augmented-student'), (1, 0, 'augmented-student'), 
# (0, 1, 'transformed-student'), (1, 0, 'transformed-student'),
# (0, 1, 'original-teacher'), (1, 0, 'original-teacher')]
# flow[0][(0, 1, 'augmented-student')] = [tf.Tensor: shape=(1, 384, 512, 2)]
def compute_loss(images, inputs):
  """Apply models and compute losses for a batch of image sequences."""
  # Check if chosen train_mode is valid.
  images = inputs.get('images')
  augmented_images = inputs.get('augmented_images', images)  # augmented_images TensorShape([1, 2, 384, 512, 3])
  full_size_images = inputs.get('full_size_images')  # full_size_images TensorShape([1, 2, 584, 912, 3])
  crop_h = inputs.get('crop_h')
  crop_w = inputs.get('crop_w')
  pad_h = inputs.get('pad_h')
  pad_w = inputs.get('pad_w')
  flows = inputs.get('flows') # [flow[0], flow[1], flow[2], flow[3], flow[4], flow[5]]


  # Compute all required flow fields.
  # TODO: Can't condition computation on this without breaking autograph.


  # Prepare occlusion estimation function.
  occlusion_estimation_fn = functools.partial(
      compute_occlusions, 
      occlusion_estimation='wang',
      occlusions_are_zeros=True,
      occ_active={'brox': False, 'wang': True},
      boundaries_occluded=full_size_images is None)

  # Prepare a simplified call for the unsupervised loss function.
  unsupervised_loss_fn = functools.partial(
      unsupervised_loss,
      weights={'supervision': 0.1, 'census': 1.0, 'smooth1': 0.0, 'smooth2': 2.0},
      occlusion_estimation_fn=occlusion_estimation_fn,only_forward=False,
      selfsup_transform_fn=build_selfsup_transformations(crop_height=64, crop_width=64, resize=True),
      fb_sigma_teacher=0.003, fb_sigma_student=0.03,
      smoothness_edge_weighting='exponential', smoothness_edge_constant=150.0,
      stop_gradient_mask=True, selfsup_mask='gaussian', smoothness_at_level=2)

  losses = {}
  sequence_unsupervised_losses = unsupervised_sequence_loss(
      images=images, flows_sequence=flows, full_size_images=full_size_images,
      crop_h=crop_h, crop_w=crop_w, pad_h=pad_h, pad_w=pad_w,
      unsupervised_loss_fn=unsupervised_loss_fn)
  losses.update(sequence_unsupervised_losses)

  losses['total'] = sum(losses.values())
  losses = {key + '-loss': losses[key] for key in losses}
  return losses

import torch
inputs = torch.load("inputs.pth")
compute_loss(inputs)