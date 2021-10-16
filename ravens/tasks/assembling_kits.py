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
# See the License for the specific language governing
# permissions and
# limitations under the License.

"""Kitting Tasks."""

import os

import numpy as np
from ravens.tasks.task import Task
from ravens.utils import utils


class AssemblingKits(Task):
    """Kitting Tasks base class."""

    def __init__(self):
        super().__init__()
        self.max_steps = 4
        self.rot_eps = np.deg2rad(30)

        # CHANGE THIS WHEN YOU ADD MORE OBJS
        self.train_set = [0,2,3,4,5,7,8,9,10]
        self.test_set = [3,7]
        self.homogeneous = False

    def reset(self, env):
        super().reset(env)

        # Add kit.
        kit_size = (0.39, 0.3, 0.0005)
        kit_urdf = 'kitting/kit.urdf'
        kit_pose = self.get_pose(env, kit_size)
        env.add_object(kit_urdf, kit_pose, 'fixed')

        if self.mode == 'train':
            n_objects = 3
            obj_shapes = np.random.choice(self.train_set, n_objects)
        else:
            if self.homogeneous:
                n_objects = 2
                obj_shapes = [np.random.choice(self.test_set)] * n_objects
            else:
                n_objects = 2
                obj_shapes = np.random.choice(self.test_set, n_objects)

        colors = [
            utils.COLORS['purple'], utils.COLORS['blue'], utils.COLORS['green'],
            utils.COLORS['yellow'], utils.COLORS['red']
        ]

        symmetry = [
            2 * np.pi, 2 * np.pi, 2 * np.pi / 3, np.pi / 2, np.pi / 2, 2 * np.pi,
            np.pi, 2 * np.pi / 5, np.pi, np.pi / 2, 2 * np.pi / 5, 0, 2 * np.pi,
            2 * np.pi, 2 * np.pi, 2 * np.pi, 0, 2 * np.pi / 6, 2 * np.pi, 2 * np.pi
        ]

        # Build kit.
        targets = []
        if self.mode == 'test':
            targ_pos = [[0, -0.05, -0.001],
                        [0, 0.05, -0.001]]
        else:
            targ_pos = [[-0.125, 0.12, -0.001],
                        [0.072, 0.07, -0.001],
                        [0.125, -0.13, -0.001]]

        template = 'kitting/object-template.urdf'
        for i in range(n_objects):
            shape = os.path.join(self.assets_root, 'kitting',
                                 f'{obj_shapes[i]:02d}.obj')
            if obj_shapes[i] == 7:
                scale = [0.006, 0.006, 0.0007]
            else:
                scale = [0.006, 0.006, 0.002]  # .0005
            pos = utils.apply(kit_pose, targ_pos[i])
            if self.mode == 'train':
                theta = np.random.rand() * 2 * np.pi
                rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
            else:
                rot = utils.eulerXYZ_to_quatXYZW((0, 0, 1.57))
            replace = {'FNAME': (shape,), 'SCALE': scale, 'COLOR': (0.2, 0.2, 0.2)}
            urdf = self.fill_template(template, replace)
            env.add_object(urdf, (pos, rot), 'fixed')
            os.remove(urdf)
            targets.append((pos, rot))

        # Add objects.
        objects = []
        matches = []
        for i in range(n_objects):
            shape = obj_shapes[i]
            size = (0.08, 0.08, 0.02)
            pose = self.get_random_pose(env, size)
            fname = f'{shape:02d}.obj'
            fname = os.path.join(self.assets_root, 'kitting', fname)
            scale = [0.006, 0.006, 0.002]
            replace = {'FNAME': (fname,), 'SCALE': scale, 'COLOR': colors[i]}
            urdf = self.fill_template(template, replace)
            block_id = env.add_object(urdf, pose)
            os.remove(urdf)
            objects.append((block_id, (symmetry[shape], None)))
            match = np.zeros(len(targets))
            match[np.argwhere(obj_shapes == shape).reshape(-1)] = 1
            matches.append(match)
        matches = np.int32(matches)
        self.goals.append((objects, matches, targets, False, True, 'pose', None, 1))

    def get_pose(self, env, kit_size):
        pose = (0.5, 0, 0.01)
        rot = (0.0, 1.57, 0, 0.9)
        return pose, rot


class AssemblingKitsEasy(AssemblingKits):
    """Kitting Task - Easy variant."""

    def __init__(self):
        super().__init__()
        self.rot_eps = np.deg2rad(30)
        self.train_set = np.int32(
            [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19])
        self.test_set = np.int32([3, 11])
        self.homogeneous = True




        # Add objects in container.
        object_points = {}
        object_ids = []
        bboxes = np.array(bboxes)
        object_template = 'box/box-template-packing.urdf'
        self.goal = {'places': {}, 'steps': []}
        for bbox in bboxes:
            size = bbox[3:] - bbox[:3]
            size[2] = size[2] - 0.01
            size[1] = size[1] - 0.01
            position = size / 2. + bbox[:3]
            position[0] += -zone_size[0] / 2
            position[1] += -zone_size[1] / 2
            position[2] += -zone_size[2] / 2
            pose = (position, (0, 0, 0, 1))
            pose = utils.multiply(zone_pose, pose)
            urdf = self.fill_template(object_template, {'DIM': size})
            box_id = env.add_object(urdf, pose)
            os.remove(urdf)
            object_ids.append((box_id, (0, None)))
            icolor = np.random.choice(range(len(colors)), 1).squeeze()
            p.changeVisualShape(box_id, -1, rgbaColor=colors[icolor] + [1])
            object_points[box_id] = self.get_object_points(box_id)

