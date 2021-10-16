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

"""Regression module."""
import tensorflow as tf


class Regression(tf.keras.Model):
    """Regression module."""

    def __init__(self):
        """Initialize a 3-layer MLP."""
        super(Regression, self).__init__()
        self.fc1 = tf.keras.layers.Dense(
            units=32,
            input_shape=(None, 1),
            kernel_initializer="normal",
            bias_initializer="normal",
            activation="relu")
        self.fc2 = tf.keras.layers.Dense(
            units=32,
            kernel_initializer="normal",
            bias_initializer="normal",
            activation="relu")
        self.fc3 = tf.keras.layers.Dense(
            units=1,
            kernel_initializer="normal",
            bias_initializer="normal")

    def __call__(self, x):
        return self.fc3(self.fc2(self.fc1(x)))

    def call(self, x):
        return self.fc3(self.fc2(self.fc1(x)))
