# coding=utf-8
# Copyright 2021 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transporter Agent (6DoF Hybrid with Regression)."""
from random import randint

from ravens import models
import tensorflow as tf
from transforms3d import quaternions
import os
import numpy as np
from ravens.tasks import cameras
from ravens.utils import utils
import matplotlib.pyplot as plt
import cv2


class Transporter6dAgent:
    """6D Transporter variant."""

    def __init__(self, name, task, root_dir, n_rotations=12):
        self.name = name
        self.task = task
        self.total_steps = 0
        self.crop_size = 64
        self.n_rotations = n_rotations
        self.pix_size = 0.003125
        self.in_shape = (320, 160, 6)
        self.in_shape_place = (95, 105, 6)
        self.cam_config = cameras.RealSenseD415.CONFIG
        self.models_dir = os.path.join(root_dir, 'checkpoints', self.name)
        self.bounds = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])
        self.bounds_place = np.array([[0.25, 0.75], [-0.5, 0.5], [-0.2, 0.8]])
        # self.bounds_place = np.array([[0.4, 0.7], [-0.5, 0.5], [0.08, 0.38]])

        self.attention_model = models.Attention(
            in_shape=self.in_shape,
            n_rotations=1,
            preprocess=utils.preprocess)
        self.transport_model = models.Transport(
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess)
        self.rpz_model = models.TransportHybrid6DoF(
            in_shape=self.in_shape,
            n_rotations=self.n_rotations,
            crop_size=self.crop_size,
            preprocess=utils.preprocess)


        self.six_dof = True
        self.p0_pixel_error = tf.keras.metrics.Mean(name='p0_pixel_error')
        self.p1_pixel_error = tf.keras.metrics.Mean(name='p1_pixel_error')
        self.p0_theta_error = tf.keras.metrics.Mean(name='p0_theta_error')
        self.p1_theta_error = tf.keras.metrics.Mean(name='p1_theta_error')
        self.metrics = [
            self.p0_pixel_error, self.p1_pixel_error, self.p0_theta_error,
            self.p1_theta_error
        ]

    def get_six_dof(self, transform_params, heightmap, pose0, pose1, augment=True):
        """Adjust SE(3) poses via the in-plane SE(2) augmentation transform."""

        p1_position, p1_rotation = pose1[0], pose1[1] #training label positions and rotations
        p0_position, p0_rotation = pose0[0], pose0[1] #training label positions and rotations

        if augment:
            t_world_center, t_world_centernew = utils.get_se3_from_image_transform(
                *transform_params, heightmap, self.bounds_place, self.pix_size)
            t_worldnew_world = t_world_centernew @ np.linalg.inv(t_world_center)
        else:
            t_worldnew_world = np.eye(4)

        p1_quat_wxyz = (p1_rotation[3], p1_rotation[0], p1_rotation[1],
                        p1_rotation[2])
        t_world_p1 = quaternions.quat2mat(p1_quat_wxyz)
        # error: index 3 is out of bounds for axis 1 with size 3
        # t_world_p1[0:3, 3] = np.array(p1_position)
        p1_position_2d = np.reshape(p1_position, (3, 1))
        t_world_p1 = np.append(t_world_p1, p1_position_2d, axis=1)
        arr = np.transpose(np.array([[0], [0], [0], [1]]))
        t_world_p1 = np.append(t_world_p1, arr, axis=0)

        t_worldnew_p1 = t_worldnew_world @ t_world_p1

        p0_quat_wxyz = (p0_rotation[3], p0_rotation[0], p0_rotation[1],
                        p0_rotation[2])

        t_world_p0 = quaternions.quat2mat(p0_quat_wxyz)
        # t_world_p0[0:3, 3] = np.array(p0_position)
        p0_position_2d = np.reshape(p0_position, (3, 1))
        t_world_p0 = np.append(t_world_p0, p0_position_2d, axis=1)
        arr = np.transpose(np.array([[0], [0], [0], [1]]))
        t_world_p0 = np.append(t_world_p0, arr, axis=0)

        t_worldnew_p0 = t_worldnew_world @ t_world_p0

        # PICK FRAME, using 0 rotation due to suction rotational symmetry
        t_worldnew_p0theta0 = t_worldnew_p0 * 1.0
        t_worldnew_p0theta0[0:3, 0:3] = np.eye(3)

        # PLACE FRAME, adjusted for this 0 rotation on pick
        t_p0_p0theta0 = np.linalg.inv(t_worldnew_p0) @ t_worldnew_p0theta0
        t_worldnew_p1theta0 = t_worldnew_p1 @ t_p0_p0theta0

        t_worldnew_p1theta0 = t_worldnew_p1theta0[0:3, 0:3]

        # convert the above rotation to euler
        quatwxyz_worldnew_p1theta0 = quaternions.mat2quat(t_worldnew_p1theta0)
        q = quatwxyz_worldnew_p1theta0
        quatxyzw_worldnew_p1theta0 = (q[1], q[2], q[3], q[0])
        p1_rotation = quatxyzw_worldnew_p1theta0
        p1_euler = utils.quatXYZW_to_eulerXYZ(p1_rotation)
        roll = p1_euler[0]
        pitch = p1_euler[1]
        p1_theta = -p1_euler[2]

        p0_theta = 0
        z = p1_position[2]


        return p0_theta, p1_theta, z, roll, pitch

    def get_image(self, obs):
        """Stack color and height images image."""

        # Get color and height maps from RGB-D images.
        cmap, hmap = utils.get_fused_heightmap(
            obs, self.cam_config, self.bounds, self.pix_size)

        # data = obs['color']
        # frame=data[0]
        #
        # plt.imshow(cmap)
        # plt.show()
        # plt.imshow(hmap)
        # plt.show()

        img = np.concatenate((cmap,
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None],
                              hmap[Ellipsis, None]), axis=2)
        assert img.shape == self.in_shape, img.shape
        return img

    def get_image_place(self, obs):
        """Stack color and height images image."""

        # Get color and height maps from RGB-D images.
        cmap_place, hmap_place = utils.get_sideways_fused_heightmap(
            obs, self.cam_config, self.bounds_place, self.pix_size)

        # cmap_place = cmap_place[135:230, 15:120]
        # hmap_place = hmap_place[135:230, 15:120]
        # plt.imshow(cmap_place)
        # plt.show()

        cmap_place = cv2.resize(cmap_place, (160, 320))
        hmap_place = cv2.resize(hmap_place, (160, 320))

        # data = obs['color']
        # frame=data[2]
        # #
        # plt.imshow(frame)
        # plt.show()
        # plt.imshow(cmap_place)
        # plt.show()
        # plt.imshow(hmap_place)
        # plt.show()

        img_place = np.concatenate((cmap_place,
                                    hmap_place[Ellipsis, None],
                                    hmap_place[Ellipsis, None],
                                    hmap_place[Ellipsis, None]), axis=2)
        assert img_place.shape == self.in_shape, img_place.shape
        return img_place

    def get_sample(self, dataset, augment=True):
        (obs, act, _, _), _ = dataset.sample()  # gets the demo image

        img_before = self.get_image(obs)  # converts the image to heightmaps and colormaps
        img_place_before = self.get_image_place(obs)

        # Get training labels from data sample.
        p0_xyz, p0_xyzw = act['pose0']
        p1_xyz, p1_xyzw = act['pose1']
        p0 = utils.xyz_to_pix(p0_xyz, self.bounds, self.pix_size)
        p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
        p1 = utils.xyz_to_pix(p1_xyz, self.bounds_place, self.pix_size)
        p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
        p1_theta = p1_theta - p0_theta
        p0_theta = 0

        if augment:
            img, _, (p0, p1), transforms = utils.perturb(img_before, [p0, p1])
            img_place, _, (p0_place, p1_place), transforms_place = utils.perturb(img_place_before, [p0, p1])
            p0_theta_place, p1_theta_place, z_place, roll_place, pitch_place = self.get_six_dof(
                transforms_place, img_place[:, :, 3], (p0_xyz, p0_xyzw), (p1_xyz, p1_xyzw))
            return img, img_place, p0, p0_theta_place, p1_place, p1_theta_place, z_place, roll_place, pitch_place

        return img_before, img_place_before, p0, p0_theta, p1, p1_theta

    def train(self, dataset, writer, num_iter=500):
        """Train on dataset for a specific number of iterations.
        Args:
          dataset: a ravens.Dataset.
          num_iter: int, number of iterations to train.
          writer: a TF summary writer (for tensorboard).
        """
        validation_rate = 1

        for i in range(num_iter):
            tf.keras.backend.set_learning_phase(1)

            #gets the labels that the model neeeds to learn
            img, img_place, p0, p0_theta, p1, p1_theta, z, roll, pitch = self.get_sample(dataset)

            #self.visualize_images(img, img_place)

            #self.visualize_images(img_place_before, img_place)

            #roll_place = 1.57

            #self.visualize_images(img, img_place)

            # Compute training losses.
            loss0 = self.attention_model.train(img, p0, p0_theta)

            loss1 = self.transport_model.train(img_place, p0, p1, p1_theta, z)

            loss2 = self.rpz_model.train(img_place, p0, p1, p1_theta, z, roll, pitch)

            with writer.as_default():
                tf.summary.scalar(
                    'attention_loss',
                    self.attention_model.metric.result(),
                    step=self.total_steps + i)
                tf.summary.scalar(
                    'transport_loss',
                    self.transport_model.metric.result(),
                    step=self.total_steps + i)
                tf.summary.scalar(
                    'z_loss',
                    self.rpz_model.z_metric.result(),
                    step=self.total_steps + i)
                tf.summary.scalar(
                    'roll_loss',
                    self.rpz_model.roll_metric.result(),
                    step=self.total_steps + i)
                tf.summary.scalar(
                    'pitch_loss',
                    self.rpz_model.pitch_metric.result(),
                    step=self.total_steps + i)

            print(f'Train Iter: {self.total_steps + i} Loss: {loss0:.2f} {loss1:.2f}  {loss2:.2f}')
        self.total_steps += num_iter
        self.save()

    def act(self, obs, info, gt_act=None):
        """Run inference and return best action given visual observations."""
        tf.keras.backend.set_learning_phase(0)

        # Get heightmap from RGB-D images.
        img = self.get_image(obs)
        img_place = self.get_image_place(obs)

        ffcmap_place = img_place[135:230, 15:120]

        self.visualize_images(img,ffcmap_place)

        # Attention model forward pass.
        attention = self.attention_model.forward(img)
        argmax = np.argmax(attention)
        argmax = np.unravel_index(argmax, shape=attention.shape)
        p0_pixel = argmax[:2]
        p0_theta = argmax[2] * (2 * np.pi / attention.shape[2])

        # Transport model forward pass.
        transport = self.transport_model.forward(img_place, p0_pixel)
        _, z, roll, pitch = self.rpz_model.forward(img_place, p0_pixel)

        argmax = np.argmax(transport)
        argmax = np.unravel_index(argmax, shape=transport.shape)

        # Index into 3D discrete tensor, grab z, roll, pitch activations
        z_best = z[:, argmax[0], argmax[1], argmax[2]][Ellipsis, None]
        roll_best = roll[:, argmax[0], argmax[1], argmax[2]][Ellipsis, None]
        pitch_best = pitch[:, argmax[0], argmax[1], argmax[2]][Ellipsis, None]

        # Send through regressors for each of z, roll, pitch
        z_best = self.rpz_model.z_regressor(z_best)[0, 0]
        roll_best = self.rpz_model.roll_regressor(roll_best)[0, 0]
        pitch_best = self.rpz_model.pitch_regressor(pitch_best)[0, 0]

        p1_pixel = argmax[:2]
        p1_theta = argmax[2] * (2 * np.pi / transport.shape[2])

        # Pixels to end effector poses.
        hmap = img[:, :, 3]
        hmap_place = img_place[:, :, 3]
        p0_position = utils.pix_to_xyz(p0_pixel, hmap, self.bounds,
                                       self.pix_size)
        p1_position = utils.pix_to_xyz(p1_pixel, hmap_place, self.bounds_place,
                                       self.pix_size)

        # value = randint(-5, 5)
        # z_best = z_best + (value/100)

        p1_position = (p1_position[0], p1_position[1], z_best)
        p0_rotation = utils.eulerXYZ_to_quatXYZW((0, 0, -p0_theta))

        p1_rotation = utils.eulerXYZ_to_quatXYZW(
            (roll_best, pitch_best, -p1_theta))

        return {
            'pose0': (np.asarray(p0_position), np.asarray(p0_rotation)),
            'pose1': (np.asarray(p1_position), np.asarray(p1_rotation))
        }

    def load(self, n_iter):
        """Load pre-trained models."""
        print(f'Loading pre-trained model at {n_iter} iterations.')
        attention_fname = 'attention-ckpt-%d.h5' % n_iter
        transport_fname = 'transport-ckpt-%d.h5' % n_iter
        rpz_fname = 'rpz-ckpt-%d.h5' % n_iter
        attention_fname = os.path.join(self.models_dir, attention_fname)
        transport_fname = os.path.join(self.models_dir, transport_fname)

        pitch_fname = os.path.join(self.models_dir, "pitch"+rpz_fname)
        z_fname = os.path.join(self.models_dir, "z" + rpz_fname)
        roll_fname = os.path.join(self.models_dir, "roll" + rpz_fname)
        rpz_fname = os.path.join(self.models_dir, rpz_fname)

        self.attention_model.load(attention_fname)
        self.transport_model.load(transport_fname)
        self.rpz_model.load(rpz_fname, pitch_fname, z_fname, roll_fname)
        self.total_iter = n_iter

    def save(self):
        """Save models."""
        if not tf.io.gfile.exists(self.models_dir):
            tf.io.gfile.makedirs(self.models_dir)
        attention_fname = 'attention-ckpt-%d.h5' % self.total_steps
        transport_fname = 'transport-ckpt-%d.h5' % self.total_steps
        rpz_fname = 'rpz-ckpt-%d.h5' % self.total_steps
        attention_fname = os.path.join(self.models_dir, attention_fname)
        transport_fname = os.path.join(self.models_dir, transport_fname)

        pitch_fname = os.path.join(self.models_dir, "pitch"+rpz_fname)
        z_fname = os.path.join(self.models_dir, "z" + rpz_fname)
        roll_fname = os.path.join(self.models_dir, "roll" + rpz_fname)
        rpz_fname = os.path.join(self.models_dir, rpz_fname)

        self.attention_model.save(attention_fname)
        self.transport_model.save(transport_fname)
        self.rpz_model.save(rpz_fname, pitch_fname, z_fname, roll_fname)

    def validate(self, test_dataset, writer):
        print('Validating!')
        tf.keras.backend.set_learning_phase(0)
        n_iter = 200
        loss0, loss1, loss2 = 0, 0, 0
        for i in range(n_iter):
            img, img_place, p0, p0_theta, p1, p1_theta, z, roll, pitch = self.get_sample\
                (test_dataset, augment=False)

            # Get validation losses. Do not backpropagate.
            loss0 += self.attention_model.train(img, p0, p0_theta, backprop=False)
            loss1 += self.transport_model.train(img_place, p0, p1, p1_theta, backprop=False)
            loss2 += self.rpz_model.train(img_place, p0, p1, p1_theta, z, roll, pitch, backprop=False)

            loss0 /= n_iter
            loss1 /= n_iter

            with writer.as_default():
                sc = tf.summary.scalar
                sc('test_loss/attention', loss0, self.total_steps)
                sc('test_loss/transport', loss1, self.total_steps)
                sc('test_loss/rpz', loss2, self.total_steps)
            print(f'Validation Loss: {loss0:.4f} {loss1:.4f} {loss2:.4f}')


    def visualize_images(self, img, img_place):
        plt.subplots(1, 2, figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(np.array(img[:, :, 0:3]).astype(np.uint8))
        plt.subplot(1, 2, 2)
        plt.imshow(np.array(img_place[:, :, 0:3]).astype(np.uint8))
        plt.tight_layout()
        plt.show()
