"""Waymo open dataset decoder.
    Taken from https://github.com/WangYueFt/pillar-od
    # Copyright (c) Massachusetts Institute of Technology and its affiliates.
    # Licensed under MIT License
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import zlib
import numpy as np

import tensorflow as tf
from pyquaternion import Quaternion

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import frame_utils

from scipy.interpolate import griddata

tf.enable_v2_behavior()

def decode_frame(frame, frame_id):
  """Decodes native waymo Frame proto to tf.Examples."""

  lidars = extract_points(frame.lasers,
                          frame.context.laser_calibrations,
                          frame.pose)

  frame_name = '{scene_name}_{location}_{time_of_day}_{timestamp}'.format(
      scene_name=frame.context.name,
      location=frame.context.stats.location,
      time_of_day=frame.context.stats.time_of_day,
      timestamp=frame.timestamp_micros)

  example_data = {
      'scene_name': frame.context.name,
      'frame_name': frame_name,
      'frame_id': frame_id,
      'lidars': lidars,
      'images': extract_images(frame),
      'camera_calibrations': extract_calibration(frame),
  }


  return example_data
  # return encode_tf_example(example_data, FEATURE_SPEC)

def decode_annos(frame, frame_id):
  """Decodes some meta data (e.g. calibration matrices, frame matrices)."""

  veh_to_global = np.array(frame.pose.transform)

  ref_pose = np.reshape(np.array(frame.pose.transform), [4, 4])
  global_from_ref_rotation = ref_pose[:3, :3] 
  objects = extract_objects(frame.laser_labels, global_from_ref_rotation)

  frame_name = '{scene_name}_{location}_{time_of_day}_{timestamp}'.format(
      scene_name=frame.context.name,
      location=frame.context.stats.location,
      time_of_day=frame.context.stats.time_of_day,
      timestamp=frame.timestamp_micros)

  annos = {
    'scene_name': frame.context.name,
    'frame_name': frame_name,
    'frame_id': frame_id,
    'veh_to_global': veh_to_global,  
    'objects': objects,
    'camera_labels': extract_boxes(frame),
    'depth_map': extract_depth_map(frame),
  }

  return annos 


def extract_points_from_range_image(laser, calibration, frame_pose):
  """Decode points from lidar."""
  if laser.name != calibration.name:
    raise ValueError('Laser and calibration do not match')
  if laser.name == dataset_pb2.LaserName.TOP:
    frame_pose = tf.convert_to_tensor(
        np.reshape(np.array(frame_pose.transform), [4, 4]))
    range_image_top_pose = dataset_pb2.MatrixFloat.FromString(
        zlib.decompress(laser.ri_return1.range_image_pose_compressed))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0],
        range_image_top_pose_tensor[..., 1], range_image_top_pose_tensor[...,
                                                                         2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[...,
                                                                          3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    frame_pose = tf.expand_dims(frame_pose, axis=0)
    pixel_pose = tf.expand_dims(range_image_top_pose_tensor, axis=0)
  else:
    pixel_pose = None
    frame_pose = None
  first_return = zlib.decompress(
      laser.ri_return1.range_image_compressed)
  second_return = zlib.decompress(
      laser.ri_return2.range_image_compressed)
  points_list = []
  for range_image_str in [first_return, second_return]:
    range_image = dataset_pb2.MatrixFloat.FromString(range_image_str)
    if not calibration.beam_inclinations:
      beam_inclinations = range_image_utils.compute_inclination(
          tf.constant([
              calibration.beam_inclination_min, calibration.beam_inclination_max
          ]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tf.constant(calibration.beam_inclinations)
    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(calibration.extrinsic.transform), [4, 4])
    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(range_image.data), range_image.shape.dims)
    range_image_mask = range_image_tensor[..., 0] > 0
    range_image_cartesian = (
        range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose,
            frame_pose=frame_pose))
    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
    points_tensor = tf.gather_nd(
        tf.concat([range_image_cartesian, range_image_tensor[..., 1:4]],
                  axis=-1),
        tf.where(range_image_mask))
    points_list.append(points_tensor.numpy())
  return points_list


def extract_points(lasers, laser_calibrations, frame_pose):
  """Extract point clouds."""
  sort_lambda = lambda x: x.name
  lasers_with_calibration = zip(
      sorted(lasers, key=sort_lambda),
      sorted(laser_calibrations, key=sort_lambda))
  points_xyz = []
  points_feature = []
  points_nlz = []
  for laser, calibration in lasers_with_calibration:
    points_list = extract_points_from_range_image(laser, calibration,
                                                  frame_pose)
    points = np.concatenate(points_list, axis=0)
    points_xyz.extend(points[..., :3].astype(np.float32))
    points_feature.extend(points[..., 3:5].astype(np.float32))
    points_nlz.extend(points[..., 5].astype(np.float32))
  return {
      'points_xyz': np.asarray(points_xyz),
      'points_feature': np.asarray(points_feature),
  }

def global_vel_to_ref(vel, global_from_ref_rotation):
  # inverse means ref_from_global, rotation_matrix for normalization
  vel = [vel[0], vel[1], 0]
  ref = np.dot(Quaternion(matrix=global_from_ref_rotation).inverse.rotation_matrix, vel) 
  ref = [ref[0], ref[1], 0.0]

  return ref

def extract_objects(laser_labels, global_from_ref_rotation):
  """Extract objects."""
  objects = []
  for object_id, label in enumerate(laser_labels):
    category_label = label.type
    box = label.box

    speed = [label.metadata.speed_x, label.metadata.speed_y]
    accel = [label.metadata.accel_x, label.metadata.accel_y]
    num_lidar_points_in_box = label.num_lidar_points_in_box
    # Difficulty level is 0 if labeler did not say this was LEVEL_2.
    # Set difficulty level of "999" for boxes with no points in box.
    if num_lidar_points_in_box <= 0:
      combined_difficulty_level = 999
    if label.detection_difficulty_level == 0:
      # Use points in box to compute difficulty level.
      if num_lidar_points_in_box >= 5:
        combined_difficulty_level = 1
      else:
        combined_difficulty_level = 2
    else:
      combined_difficulty_level = label.detection_difficulty_level

    ref_velocity = global_vel_to_ref(speed, global_from_ref_rotation)

    objects.append({
        'id': object_id,
        'name': label.id,
        'label': category_label,
        'box': np.array([box.center_x, box.center_y, box.center_z,
                         box.length, box.width, box.height, ref_velocity[0], 
                         ref_velocity[1], box.heading], dtype=np.float32),
        'num_points':
            num_lidar_points_in_box,
        'detection_difficulty_level':
            label.detection_difficulty_level,
        'combined_difficulty_level':
            combined_difficulty_level,
        'global_speed':
            np.array(speed, dtype=np.float32),
        'global_accel':
            np.array(accel, dtype=np.float32),
    })
  return objects

def extract_boxes(frame):
  """Extract bounding box labels from images
        boxes: x,y,w,h
        types: class ID
        id: Object ID
  """
  
  objects = {}
  
  for c in frame.camera_labels:
    cam_name_str = dataset_pb2.CameraName.Name.Name(c.name)
    boxes = []
    types = []
    ids = []
    
    for l in c.labels:
      ids.append(l.id)
      types.append(l.type)
      boxes.append([l.box.center_x,l.box.center_y,l.box.width,l.box.length])
    objects[cam_name_str] = {'box':boxes,'type':types,'id':ids}
  return objects

def extract_images(frame):
  """Extract camera images from frame"""
  
  
  # Save the H x W x 3 RGB image for each camera, extracted from JPEG.
  for im in frame.images:
    if dataset_pb2.CameraName.Name.Name(im.name) == 'FRONT':
      return tf.io.decode_jpeg(im.image).numpy()

  return []

def extract_calibration(frame):
  """Extract camera intrinsic and extrinsic matrices"""

  data_dict = {}

  for c in frame.context.camera_calibrations:
    cam_name_str = dataset_pb2.CameraName.Name.Name(c.name)
    data_dict[f'{cam_name_str}_INTRINSIC'] = np.array(c.intrinsic, np.float32)
    data_dict[f'{cam_name_str}_EXTRINSIC'] = np.reshape(
        np.array(c.extrinsic.transform, np.float32), [4, 4])

  return data_dict


def convert_frame_to_dict(frame):
  """Convert the frame proto into a dict of numpy arrays.
  The keys, shapes, and data types are:
    POSE: 4x4 float32 array
    TIMESTAMP: int64 scalar
    For each lidar:
      <LIDAR_NAME>_BEAM_INCLINATION: H float32 array
      <LIDAR_NAME>_LIDAR_EXTRINSIC: 4x4 float32 array
      <LIDAR_NAME>_RANGE_IMAGE_FIRST_RETURN: HxWx6 float32 array
      <LIDAR_NAME>_RANGE_IMAGE_SECOND_RETURN: HxWx6 float32 array
      <LIDAR_NAME>_CAM_PROJ_FIRST_RETURN: HxWx6 int64 array
      <LIDAR_NAME>_CAM_PROJ_SECOND_RETURN: HxWx6 float32 array
      (top lidar only) TOP_RANGE_IMAGE_POSE: HxWx6 float32 array
    For each camera:
      <CAMERA_NAME>_IMAGE: HxWx3 uint8 array
      <CAMERA_NAME>_INTRINSIC: 9 float32 array
      <CAMERA_NAME>_EXTRINSIC: 4x4 float32 array
      <CAMERA_NAME>_WIDTH: int64 scalar
      <CAMERA_NAME>_HEIGHT: int64 scalar
      <CAMERA_NAME>_SDC_VELOCITY: 6 float32 array
      <CAMERA_NAME>_POSE: 4x4 float32 array
      <CAMERA_NAME>_POSE_TIMESTAMP: float32 scalar
      <CAMERA_NAME>_ROLLING_SHUTTER_DURATION: float32 scalar
      <CAMERA_NAME>_ROLLING_SHUTTER_DIRECTION: int64 scalar
      <CAMERA_NAME>_CAMERA_TRIGGER_TIME: float32 scalar
      <CAMERA_NAME>_CAMERA_READOUT_DONE_TIME: float32 scalar
  NOTE: This function only works in eager mode for now.
  See the LaserName.Name and CameraName.Name enums in dataset.proto for the
  valid lidar and camera name strings that will be present in the returned
  dictionaries.
  Args:
    frame: open dataset frame
  Returns:
    Dict from string field name to numpy ndarray.
  """

  data_dict = {}

  # Save the H x W x 3 RGB image for each camera, extracted from JPEG.
  for im in frame.images:
    cam_name_str = dataset_pb2.CameraName.Name.Name(im.name)
    data_dict[f'{cam_name_str}_IMAGE'] = tf.io.decode_jpeg(im.image).numpy()
    data_dict[f'{cam_name_str}_SDC_VELOCITY'] = np.array([
        im.velocity.v_x, im.velocity.v_y, im.velocity.v_z, im.velocity.w_x,
        im.velocity.w_y, im.velocity.w_z
    ], np.float32)
    data_dict[f'{cam_name_str}_POSE'] = np.reshape(
        np.array(im.pose.transform, np.float32), (4, 4))
    data_dict[f'{cam_name_str}_POSE_TIMESTAMP'] = np.array(
        im.pose_timestamp, np.float32)
    data_dict[f'{cam_name_str}_ROLLING_SHUTTER_DURATION'] = np.array(im.shutter)
    data_dict[f'{cam_name_str}_CAMERA_TRIGGER_TIME'] = np.array(
        im.camera_trigger_time)
    data_dict[f'{cam_name_str}_CAMERA_READOUT_DONE_TIME'] = np.array(
        im.camera_readout_done_time)

  # Save the intrinsics, 4x4 extrinsic matrix, width, and height of each camera.
  for c in frame.context.camera_calibrations:
    cam_name_str = dataset_pb2.CameraName.Name.Name(c.name)
    data_dict[f'{cam_name_str}_INTRINSIC'] = np.array(c.intrinsic, np.float32)
    data_dict[f'{cam_name_str}_EXTRINSIC'] = np.reshape(
        np.array(c.extrinsic.transform, np.float32), [4, 4])
    data_dict[f'{cam_name_str}_WIDTH'] = np.array(c.width)
    data_dict[f'{cam_name_str}_HEIGHT'] = np.array(c.height)
    data_dict[f'{cam_name_str}_ROLLING_SHUTTER_DIRECTION'] = np.array(
        c.rolling_shutter_direction)

  data_dict['POSE'] = np.reshape(
      np.array(frame.pose.transform, np.float32), (4, 4))
  data_dict['TIMESTAMP'] = np.array(frame.timestamp_micros)

  return data_dict


def extract_depth_map(frame):
  """
  Extract front-view lidar camera projection for ground-truth depth maps
  """
  (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

  for c in frame.context.camera_calibrations:
    if dataset_pb2.CameraName.Name.Name(c.name) == 'FRONT':
      extrinsic = np.reshape(
        np.array(c.extrinsic.transform, np.float32), [4, 4])
  range_images_cartesian = convert_range_image_to_cartesian(frame,range_images,range_image_top_pose)
  cam_projection = (np.array(camera_projections[1][0].data).reshape(64,2650,6))[np.newaxis,...]
  depth = range_image_utils.build_camera_depth_image(range_images_cartesian[1][np.newaxis,...],extrinsic[np.newaxis,...],cam_projection ,[1280,1920],1)
  p = np.where(depth[0]!= 0)
  v = np.extract(depth[0]!=0,depth[0])
  grid_w,grid_h = np.mgrid[0:1280,0:1920]
  depth_map = griddata(p, v, (grid_w, grid_h), method='nearest')
  depth_map = depth_map/np.max(depth_map)

  return depth_map[0:1280:4,0:1920:4]



def convert_range_image_to_cartesian(frame,
                                     range_images,
                                     range_image_top_pose,
                                     ri_index=0,
                                     keep_polar_features=False):
  """Convert range images from polar coordinates to Cartesian coordinates.
  Args:
    frame: open dataset frame
    range_images: A dict of {laser_name, [range_image_first_return,
       range_image_second_return]}.
    range_image_top_pose: range image pixel pose for top lidar.
    ri_index: 0 for the first return, 1 for the second return.
    keep_polar_features: If true, keep the features from the polar range image
      (i.e. range, intensity, and elongation) as the first features in the
      output range image.
  Returns:
    dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
      will be 3 if keep_polar_features is False (x, y, z) and 6 if
      keep_polar_features is True (range, intensity, elongation, x, y, z).
  """
  cartesian_range_images = {}
  frame_pose = tf.convert_to_tensor(
      value=np.reshape(np.array(frame.pose.transform), [4, 4]))

  # [H, W, 6]
  range_image_top_pose_tensor = tf.reshape(
      tf.convert_to_tensor(value=range_image_top_pose.data),
      range_image_top_pose.shape.dims)
  # [H, W, 3, 3]
  range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
      range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
      range_image_top_pose_tensor[..., 2])
  range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
  range_image_top_pose_tensor = transform_utils.get_transform(
      range_image_top_pose_tensor_rotation,
      range_image_top_pose_tensor_translation)

  for c in frame.context.laser_calibrations:
    range_image = range_images[c.name][ri_index]
    if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
      beam_inclinations = range_image_utils.compute_inclination(
          tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
          height=range_image.shape.dims[0])
    else:
      beam_inclinations = tf.constant(c.beam_inclinations)

    beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
    extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

    range_image_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
    pixel_pose_local = None
    frame_pose_local = None
    if c.name == dataset_pb2.LaserName.TOP:
      pixel_pose_local = range_image_top_pose_tensor
      pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
      frame_pose_local = tf.expand_dims(frame_pose, axis=0)
    range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
        tf.expand_dims(range_image_tensor[..., 0], axis=0),
        tf.expand_dims(extrinsic, axis=0),
        tf.expand_dims(tf.convert_to_tensor(value=beam_inclinations), axis=0),
        pixel_pose=pixel_pose_local,
        frame_pose=frame_pose_local)

    range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

    if keep_polar_features:
      # If we want to keep the polar coordinate features of range, intensity,
      # and elongation, concatenate them to be the initial dimensions of the
      # returned Cartesian range image.
      range_image_cartesian = tf.concat(
          [range_image_tensor[..., 0:3], range_image_cartesian], axis=-1)

    cartesian_range_images[c.name] = range_image_cartesian

  return cartesian_range_images