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

"""Transport 6DoF models."""

import numpy as np
from ravens.models.regression import Regression
from ravens.models.transport import Transport
import tensorflow as tf


class TransportHybrid6DoF(Transport):
    """Transport + 6DoF regression hybrid."""

    def __init__(self, in_shape, n_rotations, crop_size, preprocess):
        self.output_dim = 24
        self.kernel_dim = 24
        super().__init__(in_shape, n_rotations, crop_size, preprocess)

        self.regress_loss = tf.keras.losses.Huber()

        self.z_regressor = Regression()
        self.roll_regressor = Regression()
        self.pitch_regressor = Regression()

        self.z_metric = tf.keras.metrics.Mean(name="loss_z")
        self.roll_metric = tf.keras.metrics.Mean(name="loss_roll")
        self.pitch_metric = tf.keras.metrics.Mean(name="loss_pitch")

    def correlate(self, in0, in1, softmax):
        output = tf.nn.convolution(
            in0[Ellipsis, :3], in1[:, :, :3, :], data_format="NHWC")
        z_tensor = tf.nn.convolution(
            in0[Ellipsis, :8], in1[:, :, :8, :], data_format="NHWC")
        roll_tensor = tf.nn.convolution(
            in0[Ellipsis, 8:16], in1[:, :, 16:24, :], data_format="NHWC")
        pitch_tensor = tf.nn.convolution(
            in0[Ellipsis, 16:24], in1[:, :, 16:24, :], data_format="NHWC")
        if softmax:
            output_shape = output.shape
            output = tf.reshape(output, (1, np.prod(output.shape)))
            output = tf.nn.softmax(output)
            output = np.float32(output).reshape(output_shape[1:])
        return output, z_tensor, roll_tensor, pitch_tensor

    def train(self, in_img, p, q, theta, z, roll, pitch, backprop=True):
        self.metric.reset_states()
        self.z_metric.reset_states()
        self.roll_metric.reset_states()
        self.pitch_metric.reset_states()
        with tf.GradientTape() as tape:
            output = self.forward(in_img, p, softmax=False)
            output, z_tensor, roll_tensor, pitch_tensor = output

            # Get one-hot pixel label map and 6DoF labels.
            itheta = theta / (2 * np.pi / self.n_rotations)
            itheta = np.int32(np.round(itheta)) % self.n_rotations
            label_size = in_img.shape[:2] + (self.n_rotations,)
            label = np.zeros(label_size)
            label[q[0], q[1], itheta] = 1
            z_label, roll_label, pitch_label = z, roll, pitch

            # Use a window for regression rather than only exact.
            u_window = 7
            v_window = 7
            theta_window = 1
            u_min = max(q[0] - u_window, 0)
            u_max = min(q[0] + u_window + 1, z_tensor.shape[1])
            v_min = max(q[1] - v_window, 0)
            v_max = min(q[1] + v_window + 1, z_tensor.shape[2])
            theta_min = max(itheta - theta_window, 0)
            theta_max = min(itheta + theta_window + 1, z_tensor.shape[3])

            z_est_at_xytheta = z_tensor[0, u_min:u_max, v_min:v_max,
                               theta_min:theta_max]
            roll_est_at_xytheta = roll_tensor[0, u_min:u_max, v_min:v_max,
                                  theta_min:theta_max]
            pitch_est_at_xytheta = pitch_tensor[0, u_min:u_max, v_min:v_max,
                                   theta_min:theta_max]

            z_est_at_xytheta = tf.reshape(z_est_at_xytheta, (-1, 1))
            roll_est_at_xytheta = tf.reshape(roll_est_at_xytheta, (-1, 1))
            pitch_est_at_xytheta = tf.reshape(pitch_est_at_xytheta, (-1, 1))

            z_est_at_xytheta = self.z_regressor(z_est_at_xytheta)
            roll_est_at_xytheta = self.roll_regressor(roll_est_at_xytheta)
            pitch_est_at_xytheta = self.pitch_regressor(pitch_est_at_xytheta)

            z_weight = 10.0
            roll_weight = 10.0
            pitch_weight = 10.0

            z_label = tf.convert_to_tensor(z_label)[None, Ellipsis]
            roll_label = tf.convert_to_tensor(roll_label)[None, Ellipsis]
            pitch_label = tf.convert_to_tensor(pitch_label)[None, Ellipsis]

            z_loss = z_weight * self.regress_loss(z_label, z_est_at_xytheta)
            roll_loss = roll_weight * self.regress_loss(roll_label,
                                                        roll_est_at_xytheta)
            pitch_loss = pitch_weight * self.regress_loss(pitch_label,
                                                          pitch_est_at_xytheta)

            loss = z_loss + roll_loss + pitch_loss

            train_vars = self.model.trainable_variables + \
                         self.z_regressor.trainable_variables + \
                         self.roll_regressor.trainable_variables + \
                         self.pitch_regressor.trainable_variables

            if backprop:
                grad = tape.gradient(loss, train_vars)
                self.optim.apply_gradients(zip(grad, train_vars))

            self.z_metric(z_loss)
            self.roll_metric(roll_loss)
            self.pitch_metric(pitch_loss)

        self.iters += 1
        return np.float32(loss)

    def save(self, fname, pitch_fname, z_fname, roll_fname):
        self.model.save(fname)
        self.z_regressor.save_weights(z_fname)
        self.pitch_regressor.save_weights(pitch_fname)
        self.roll_regressor.save_weights(roll_fname)

    def load(self, fname, pitch_fname, z_fname, roll_fname):
        self.model.load_weights(fname)
        self.z_regressor = Regression()
        self.pitch_regressor = Regression()
        self.roll_regressor = Regression()

        self.z_regressor.build(input_shape=(None, 1))
        self.pitch_regressor.build(input_shape=(None, 1))
        self.roll_regressor.build(input_shape=(None, 1))

        self.z_regressor.summary()
        self.z_regressor.load_weights(z_fname)
        self.pitch_regressor.load_weights(pitch_fname)
        self.roll_regressor.load_weights(roll_fname)

