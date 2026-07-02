import h5py
import os
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import argparse
import glob
import re

class ReadH5Files():
    def __init__(self, robot_infor):
        self.camera_names = robot_infor['camera_names']
        self.camera_sensors = robot_infor['camera_sensors']

        self.arms = robot_infor['arms']
        self.robot_infor = robot_infor['controls']
        self.default_demo = robot_infor.get('default_demo', None)

    def decoder_image(self, camera_rgb_images, camera_depth_images):
        if type(camera_rgb_images[0]) is np.uint8:
            rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = np.asarray([])
            
            return rgb, depth
        else:
            rgb_images = []
            depth_images = []
            for idx, camera_rgb_image in enumerate(camera_rgb_images):
                rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
                if camera_depth_images is not None:
                    depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                    depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
                else:
                    depth = np.asarray([])
                rgb_images.append(rgb)
                depth_images.append(depth)
                
            rgb_images = np.asarray(rgb_images)
            depth_images = np.asarray(depth_images)
            return rgb_images, depth_images
        
    def read_aug_images(self, aug_path):
        aug_images = []
        
        # Extract the index from the file name using a regular expression
        def extract_number(file_name):
            match = re.search(r'_(\d+)\.png$', file_name)
            if match:
                return int(match.group(1))
            else:
                return -1  # return -1 if no index is matched
        
        # Get the file name list and sort it by index
        image_names = sorted(os.listdir(aug_path), key=extract_number)
        
        for image_name in image_names:
            if image_name.endswith('.png'):
                image_path = os.path.join(aug_path, image_name)
                aug_image = cv2.imread(image_path)
                if aug_image is None:
                    print(f"Failed to read image: {image_path}")
                    continue
                aug_image = cv2.imencode(".jpg", aug_image)[1]
                aug_images.append(aug_image)
        
        return aug_images
    
    def read_aug_images_matrix(self, image_list):
        aug_images_list= []
        for image in image_list:
            aug_image = cv2.imencode(".jpg", image)[1]
            aug_image = np.asarray(aug_image)
            aug_images_list.append(aug_image)
        return aug_images_list

    def create_new_h5_file(self, src_root, new_file_path):
        # Create a new HDF5 file and copy the original file's structure
        with h5py.File(new_file_path, 'w') as dst_root:
            def copy_group(name, obj):
                if isinstance(obj, h5py.Group):
                    dst_root.create_group(name)
                elif isinstance(obj, h5py.Dataset):
                    try:
                        # Handle scalar datasets
                        if obj.shape == ():
                            dst_root.create_dataset(name, data=obj[()])
                        else:
                            dst_root.create_dataset(name, data=obj[:])
                    except Exception as e:
                        print(f"Warning: Failed to copy dataset {name}: {e}")

            src_root.visititems(copy_group)

            # Copy attributes
            for attr_key in src_root.attrs:
                dst_root.attrs[attr_key] = src_root.attrs[attr_key]
        

    def _load_new_format(self, root, camera_frame=None):
        """
        Handle files structured as:
        data/demo_*/obs/{agentview_rgb, eye_in_hand_rgb, ...}
        """
        if 'data' not in root:
            return None  # not the new format

        data_group = root['data']
        # pick demo
        if self.default_demo is not None and self.default_demo in data_group:
            demo_key = self.default_demo
        else:
            demo_keys = sorted(k for k in data_group.keys() if k.startswith('demo_'))
            if not demo_keys:
                return None
            demo_key = demo_keys[0]

        obs_group = data_group[demo_key]['obs']

        rgb_images_dict = {}
        depth_images_dict = {}
        masked_rgb_dict = defaultdict(dict)

        for cam_name in self.camera_names:
            if cam_name not in obs_group:
                print(f"Warning: camera '{cam_name}' not found in {demo_key}/obs")
                continue
            ds = obs_group[cam_name]
            if camera_frame is not None:
                rgb = ds[camera_frame]
                if rgb.ndim == 3:
                    rgb = np.expand_dims(rgb, 0)
            else:
                rgb = ds[:]
            rgb_images_dict[cam_name] = np.asarray(rgb[:, ::-1])
            depth_images_dict[cam_name] = np.zeros_like(rgb)

            # Load masked RGB images (new format)
            masked_sources = []
            if 'masked_images' in data_group[demo_key]:
                masked_sources.append(data_group[demo_key]['masked_images'])
            if 'masked_images' in root:
                masked_sources.append(root['masked_images'])

            for masked_root in masked_sources:
                if cam_name not in masked_root:
                    continue
                cam_group = masked_root[cam_name]
                for label in cam_group.keys():
                    ds = cam_group[label]
                    masked = ds[camera_frame] if camera_frame is not None else ds[:]
                    if isinstance(masked, np.ndarray):
                        if masked.ndim == 4:
                            masked_rgb_dict[cam_name][label] = masked
                        elif masked.ndim == 3:
                            masked_rgb_dict[cam_name][label] = np.expand_dims(masked, 0)
                break  # prefer the first available masked_images source

        return rgb_images_dict, depth_images_dict, masked_rgb_dict

    def execute(self, file_path, camera_frame=None, control_frame=None):
        rgb_images_dict = {}
        depth_images_dict = {}
        masked_rgb_dict = defaultdict(dict)
        
        with h5py.File(file_path, 'r') as root:
            # Try new-format loader first
            new_fmt = self._load_new_format(root, camera_frame)
            if new_fmt is not None:
                return new_fmt

            for cam_name in self.camera_names:
                if 'observations' in root:
                    if camera_frame is not None:
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][camera_frame],
                            camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][camera_frame])
                    else:
                        decode_rgb, decode_depth = self.decoder_image(
                            camera_rgb_images=root['observations'][self.camera_sensors[0]][cam_name][:],
                            camera_depth_images=root['observations'][self.camera_sensors[1]][cam_name][:])
                    
                    rgb_images_dict[cam_name] = decode_rgb
                    depth_images_dict[cam_name] = decode_depth
                
                # Read masked RGB images if they exist
                if 'masked_images' in root:
                    if cam_name in root['masked_images']:
                        cam_group = root['masked_images'][cam_name]
                        for label in cam_group.keys():
                            try:
                                if camera_frame is not None:
                                # Read the masked image of a specific frame
                                    masked_rgb = cam_group[label][camera_frame]
                                else:
                                    # Directly read the masked RGB data
                                    masked_rgb = cam_group[label][:]
                                print(f'masked shape:{masked_rgb.shape}')
                                if isinstance(masked_rgb, np.ndarray):
                                    if len(masked_rgb.shape) == 4:  # Multiple frames
                                        masked_rgb_dict[cam_name][label] = masked_rgb
                                    elif len(masked_rgb.shape) == 3:  # Single frame
                                        masked_rgb_dict[cam_name][label] = np.expand_dims(masked_rgb, 0)
                                
                                else:
                                    print(f"Warning: Unexpected data type for masked RGB: {type(masked_rgb)}")
                            except Exception as e:
                                print(f"Error reading masked RGB for camera {cam_name}, label {label}: {e}")
                                continue
        
        return rgb_images_dict, depth_images_dict, masked_rgb_dict
    
    def check_original_file(self, file_path, cam_name):
        with h5py.File(file_path, 'r') as src_root:
            print(f"Original file dataset shape: {src_root['observations'][self.camera_sensors[0]][cam_name].shape}")

    def _select_demo_key(self, data_group):
        """Pick a demo key based on default_demo or the first sorted demo_* key."""
        if self.default_demo is not None and self.default_demo in data_group:
            return self.default_demo
        demo_keys = sorted(k for k in data_group.keys() if k.startswith('demo_'))
        return demo_keys[0] if demo_keys else None
    
    def write_aug_images_to_h5(self, new_file_path, cam_name, aug_rgb_images, camera_frame=None):
        # Write the augmented image data into the new HDF5 file
        with h5py.File(new_file_path, 'r+') as dst_root:
            is_new_format = 'data' in dst_root

            if is_new_format:
                demo_key = self._select_demo_key(dst_root['data'])
                if demo_key is None:
                    raise KeyError("No demo_* group found under 'data'.")
                if cam_name not in dst_root['data'][demo_key]['obs']:
                    raise KeyError(f"Camera '{cam_name}' not found in data/{demo_key}/obs")

                ds = dst_root['data'][demo_key]['obs'][cam_name]  # shape: (T, H, W, 3), dtype uint8

                # decode and write
                for i, enc_img in enumerate(aug_rgb_images):
                    rgb_bgr = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
                    if rgb_bgr is None:
                        raise ValueError(f"Failed to decode augmented image index {i} for camera {cam_name}")
                    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

                    write_idx = (camera_frame + i) if camera_frame is not None else i
                    if write_idx >= ds.shape[0]:
                        raise IndexError(f"write index {write_idx} exceeds dataset length {ds.shape[0]}")
                    if rgb.shape != ds.shape[1:]:
                        raise ValueError(f"Shape mismatch: decoded {rgb.shape}, dataset expects {ds.shape[1:]}")
                    ds[write_idx] = rgb
            else:
                # legacy layout
                dataset = dst_root['observations'][self.camera_sensors[0]][cam_name]
                for i, enc_img in enumerate(aug_rgb_images):
                    write_idx = i if camera_frame is None else camera_frame + i
                    rgb = enc_img
                    # keep legacy behavior: aug_rgb_images already encoded bytes; write directly
                    dataset[write_idx] = rgb

    def write_masks_to_h5(self, h5_file_path, masks_dict):
        """
        Write mask data into an h5 file.
        
        Args:
            h5_file_path: path to the h5 file
            masks_dict: dictionary holding the mask data, formatted as:
                        {camera_name: {label_name: mask_array}}
        """
        with h5py.File(h5_file_path, 'r+') as h5_file:
            # Create the masks group if it does not exist
            if 'masks' not in h5_file:
                masks_group = h5_file.create_group('masks')
            else:
                masks_group = h5_file['masks']
                
            # Iterate over each camera
            for cam_name, labels_dict in masks_dict.items():
                # Create a group for each camera
                if cam_name not in masks_group:
                    cam_group = masks_group.create_group(cam_name)
                else:
                    cam_group = masks_group[cam_name]
                    
                # Iterate over the mask of each label
                for label_name, mask_array in labels_dict.items():
                    # Save the mask data as a dataset
                    if label_name in cam_group:
                        del cam_group[label_name]  # delete it if it already exists
                    
                    # Save the mask array
                    cam_group.create_dataset(label_name, data=mask_array, compression="gzip")

    def write_masks_and_masked_images_to_h5(self, h5_file_path, masks_dict, masked_images_dict):
        with h5py.File(h5_file_path, 'r+') as h5_file:
            # Create masks group
            if 'masks' not in h5_file:
                masks_group = h5_file.create_group('masks')
            else:
                masks_group = h5_file['masks']
            
            # Create masked_images group
            if 'masked_images' not in h5_file:
                masked_images_group = h5_file.create_group('masked_images')
            else:
                masked_images_group = h5_file['masked_images']
            
            # Write both masks and masked images
            for cam_name, labels_dict in masks_dict.items():
                # Handle masks
                if cam_name not in masks_group:
                    cam_group_masks = masks_group.create_group(cam_name)
                else:
                    cam_group_masks = masks_group[cam_name]
                
                # Handle masked images
                if cam_name not in masked_images_group:
                    cam_group_images = masked_images_group.create_group(cam_name)
                else:
                    cam_group_images = masked_images_group[cam_name]
                
                # Write data for each label
                for label_name, mask_array in labels_dict.items():
                    # Write mask
                    if label_name in cam_group_masks:
                        del cam_group_masks[label_name]
                    cam_group_masks.create_dataset(label_name, data=mask_array, compression="gzip")
                    
                    # Write masked image
                    if label_name in cam_group_images:
                        del cam_group_images[label_name]
                    cam_group_images.create_dataset(label_name, 
                                                 data=masked_images_dict[cam_name][label_name], 
                                                 compression="gzip")
                    
    def create_new_h5_with_masks(self, output_path, masks_dict, masked_images_dict):
        """
        Create a new H5 file with only masks and masked images.
        
        Args:
            output_path: Path for the new H5 file
            masks_dict: Dictionary of masks {camera_name: {label_name: mask_array}}
            masked_images_dict: Dictionary of masked images {camera_name: {label_name: image_array}}
        """
        with h5py.File(output_path, 'w') as h5_file:
            # Create masks group
            masks_group = h5_file.create_group('masks')
            
            # Create masked_images group
            masked_images_group = h5_file.create_group('masked_images')
            
            # Write both masks and masked images
            for cam_name, labels_dict in masks_dict.items():
                # Create camera groups
                cam_group_masks = masks_group.create_group(cam_name)
                cam_group_images = masked_images_group.create_group(cam_name)
                
                # Write data for each label
                for label_name, mask_array in labels_dict.items():
                    # Write mask
                    cam_group_masks.create_dataset(
                        label_name, 
                        data=mask_array, 
                        compression="gzip"
                    )
                    
                    # Write masked image
                    cam_group_images.create_dataset(
                        label_name, 
                        data=masked_images_dict[cam_name][label_name], 
                        compression="gzip"
                    )

    def process_aug_images(self, file_path, save_dir, output_dir, camera_frame=None):
        new_file_path = os.path.join(output_dir, 'aug_data.h5')

        # Read the original file
        with h5py.File(file_path, 'r') as src_root:
            # Create the new file and copy the structure
            self.create_new_h5_file(src_root, new_file_path)

            # Read augmented images and write them into the new HDF5 file
            for cam_name in self.camera_names:
                cam_dir = os.path.join(save_dir, cam_name)
                if os.path.exists(cam_dir) and os.path.isdir(cam_dir):
                    run_dirs = [d for d in os.listdir(cam_dir) if d.startswith('run_')]
                    for run_dir in run_dirs:
                        run_id = int(run_dir.split('_')[1])
                        if camera_frame is not None and run_id != camera_frame:
                            continue
                        aug_path = os.path.join(cam_dir, run_dir)
                        aug_rgb_images = self.read_aug_images(aug_path)
                        self.write_aug_images_to_h5(new_file_path, cam_name, aug_rgb_images, run_id)

    def process_aug_images_matrix(self, file_path, new_rgb, output_dir):
        aug_path = os.path.join(output_dir, 'aug_data.h5')
        
        # Read the original file and create the new file
        with h5py.File(file_path, 'r') as src_root:
            self.create_new_h5_file(src_root, aug_path)

        # Write augmented images for each camera
        for cam_name in self.camera_names:
            if cam_name in new_rgb:  # make sure this camera has corresponding augmented data
                aug_rgb_images = self.read_aug_images_matrix(new_rgb[cam_name])
                self.write_aug_images_to_h5(aug_path, cam_name, aug_rgb_images)
                del aug_rgb_images

    def process_masks_and_masked_images(self, masks_dict, masked_images_dict, output_dir):
        mask_path = os.path.join(output_dir, 'masks.h5')            
        if masks_dict is not None and masked_images_dict is not None:
            self.create_new_h5_with_masks(mask_path, masks_dict, masked_images_dict)

    def save_images_to_local(self, save_dir, rgb_images_dict, masked_rgb_dict, depth_images_dict):
        """
        Save all types of images to local directory
        
        Args:
            save_dir: Base directory to save images
            rgb_images_dict: Dictionary of RGB images by camera
            masked_rgb_dict: Dictionary of masked RGB images by camera and label
            depth_images_dict: Dictionary of depth images by camera
        """
        for cam_name in self.camera_names:
            # Create camera directory
            cam_dir = os.path.join(save_dir, cam_name)
            os.makedirs(cam_dir, exist_ok=True)
            
            ##Save RGB images
            if cam_name in rgb_images_dict.keys():
                rgb_dir = os.path.join(cam_dir, 'rgb')
                os.makedirs(rgb_dir, exist_ok=True)
                for i, rgb_img in enumerate(rgb_images_dict[cam_name]):
                    rgb_path = os.path.join(rgb_dir, f'frame_{i:04d}.png')
                    cv2.imwrite(rgb_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))

            # Save masked RGB images
            if cam_name in masked_rgb_dict:
                masked_dir = os.path.join(cam_dir, 'masked')
                os.makedirs(masked_dir, exist_ok=True)
                for label, masked_imgs in masked_rgb_dict[cam_name].items():
                    label_dir = os.path.join(masked_dir, label)
                    os.makedirs(label_dir, exist_ok=True)
                    for i, masked_img in enumerate(masked_imgs):
                        masked_path = os.path.join(label_dir, f'frame_{i:04d}.png')
                        cv2.imwrite(masked_path, cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

def str_to_list(camera_names_str):
    # Split the string by commas and strip the whitespace around each element
    camera_names = [name.strip() for name in camera_names_str.split(',')]
    return camera_names

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Read and process HDF5 files.")
    parser.add_argument('--camera_names', type=str, required=True, help='Comma-separated list of camera names')
    parser.add_argument('--camera_frame', type=int, required=False, default=None)
    parser.add_argument('--file_path', type=str, required=True, help='Path to the HDF5 file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the processed data')

    args = parser.parse_args()
    camera_names = args.camera_names
    file_path = args.file_path
    save_dir = args.save_dir
    camera_frame = args.camera_frame
    camera_names = str_to_list(camera_names)

    os.makedirs(save_dir, exist_ok=True)

    robot_infor = {'camera_names': camera_names,
                   'camera_sensors': ['rgb_images','depth_images'],
                   'arms': ['master', 'puppet'],
                   'controls': ['joint_position']}

    read_h5files = ReadH5Files(robot_infor)
    rgb_images_dict, depth_images_dict, masked_rgb_dict = read_h5files.execute(file_path=file_path, camera_frame=camera_frame)
    
    # Save all images to local directory
    output_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    read_h5files.save_images_to_local(output_dir, rgb_images_dict, masked_rgb_dict, depth_images_dict)