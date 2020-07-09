from scipy import spatial
from skimage import io
import numpy as np
import numpy.ma as ma
import time
import sys
from scipy import ndimage
import matplotlib.pyplot as plt

sys.path.append(sys.path[0] + '/..')
from build.inverse_sensor_model import *
from random import shuffle
import os


class Robot:
    def __init__(self, index_map, train, plot):
        self.mode = train
        self.plot = plot
        if self.mode:
            self.map_dir = '../DungeonMaps/train'
        else:
            self.map_dir = '../DungeonMaps/test'
        self.map_list = os.listdir(self.map_dir)
        self.map_number = np.size(self.map_list)
        shuffle(self.map_list)
        self.li_map = index_map
        self.global_map, self.robot_position = self.map_setup(self.map_dir + '/' + self.map_list[self.li_map])
        self.op_map = np.ones(self.global_map.shape) * 127
        self.map_size = np.shape(self.global_map)
        self.resolution = 1
        self.sensor_range = 80
        self.old_position = np.zeros([2])
        self.old_op_map = np.empty([0])
        current_dir = os.path.dirname(os.path.realpath(__file__))
        self.action_space = np.genfromtxt(current_dir + '/action_points.csv', delimiter=",")
        self.t = self.map_points(self.global_map)
        self.free_tree = spatial.KDTree(self.free_points(self.global_map).tolist())
        self.robot_size = 6
        self.local_size = 40
        if self.plot:
            self.xPoint = []
            self.yPoint = []

    def begin(self):
        self.op_map = self.inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)
        step_map = self.robot_model(self.robot_position, self.robot_size, self.t, self.op_map)
        map_local = self.local_map(self.robot_position, step_map, self.map_size, self.sensor_range + self.local_size)
        if self.plot:
            self.xPoint.append(self.robot_position[0])
            self.yPoint.append(self.robot_position[1])
            self.gui()
        return map_local

    def step(self, action_index):
        terminal = False
        complete = False
        new_location = False
        self.old_position = self.robot_position.copy()
        self.old_op_map = self.op_map.copy()

        # take action
        self.take_action(action_index, self.robot_position)

        # collision check
        collision_points, collision_index = self.collision_check(self.old_position, self.robot_position, self.map_size,
                                                                 self.global_map)

        if collision_index:
            self.robot_position = self.nearest_free(self.free_tree, collision_points)
            self.op_map = self.inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)
            step_map = self.robot_model(self.robot_position, self.robot_size, self.t, self.op_map)
        else:
            self.op_map = self.inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)
            step_map = self.robot_model(self.robot_position, self.robot_size, self.t, self.op_map)

        map_local = self.local_map(self.robot_position, step_map, self.map_size, self.sensor_range + self.local_size)
        reward = self.get_reward(self.old_op_map, self.op_map, collision_index)

        if reward == -1:
            terminal = True
        elif reward <= 0.02:
            reward = -0.8
            new_location = True
            terminal = True

        if self.plot:
            self.xPoint.append(self.robot_position[0])
            self.yPoint.append(self.robot_position[1])
            self.gui()

        if collision_index:
            self.robot_position = self.old_position.copy()
            self.op_map = self.old_op_map.copy()
            if self.plot:
                self.xPoint.pop()
                self.yPoint.pop()
                self.gui()

        if np.size(np.where(self.global_map == 255)) - np.size(np.where(self.op_map == 255)) < 500:
            self.li_map += 1
            if self.li_map == self.map_number:
                self.li_map = 0
            self.__init__(self.li_map, self.mode, self.plot)
            complete = True
            new_location = False
            terminal = True

        return map_local, reward, terminal, complete, new_location, collision_index

    def rescuer(self):
        complete = False
        all_map = False

        self.robot_position = self.frontier(self.op_map, self.map_size, self.t)
        self.op_map = self.inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)
        step_map = self.robot_model(self.robot_position, self.robot_size, self.t, self.op_map)
        map_local = self.local_map(self.robot_position, step_map, self.map_size, self.sensor_range + self.local_size)

        if self.plot:
            self.xPoint.append(ma.masked)
            self.yPoint.append(ma.masked)
            self.xPoint.append(self.robot_position[0])
            self.yPoint.append(self.robot_position[1])
            self.gui()

        if np.size(np.where(self.global_map == 255)) - np.size(np.where(self.op_map == 255)) < 500:
            self.li_map += 1
            if self.li_map == self.map_number:
                self.li_map = 0
            self.__init__(self.li_map, self.mode, self.plot)
            complete = True
            new_location = False
            terminal = True
        return map_local, complete

    def take_action(self, action_index, robot_position):
        move_action = self.action_space[action_index, :]
        robot_position[0] = np.round(robot_position[0] + move_action[0])
        robot_position[1] = np.round(robot_position[1] + move_action[1])

    def map_setup(self, location):
        global_map = (io.imread(location, 1) * 255).astype(int)
        robot_location = np.nonzero(global_map == 208)
        robot_location = np.array([np.array(robot_location)[1, 127], np.array(robot_location)[0, 127]])
        global_map = (global_map > 150)
        global_map = global_map * 254 + 1
        return global_map, robot_location

    def map_points(self, map_glo):
        map_x = map_glo.shape[1]
        map_y = map_glo.shape[0]
        x = np.linspace(0, map_x - 1, map_x)
        y = np.linspace(0, map_y - 1, map_y)
        t1, t2 = np.meshgrid(x, y)
        points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
        return points

    def local_map(self, robot_location, map_glo, map_size, local_size):
        minX = robot_location[0] - local_size
        maxX = robot_location[0] + local_size
        minY = robot_location[1] - local_size
        maxY = robot_location[1] + local_size

        if minX < 0:
            maxX = abs(minX) + maxX
            minX = 0
        if maxX > map_size[1]:
            minX = minX - (maxX - map_size[1])
            maxX = map_size[1]
        if minY < 0:
            maxY = abs(minY) + maxY
            minY = 0
        if maxY > map_size[0]:
            minY = minY - (maxY - map_size[0])
            maxY = map_size[0]

        map_loc = map_glo[minY:maxY][:, minX:maxX]
        return map_loc

    def free_points(self, op_map):
        index = np.where(op_map == 255)
        free = np.asarray([index[1], index[0]]).T
        return free

    def get_reward(self, old_op_map, op_map, coll_index):
        if not coll_index:
            reward = float(np.size(np.where(op_map == 255)) - np.size(np.where(old_op_map == 255))) / 14000
            if reward > 1:
                reward = 1
        else:
            reward = -1
        return reward

    def nearest_free(self, tree, point):
        pts = np.atleast_2d(point)
        index = tuple(tree.query(pts)[1])
        nearest = tree.data[index]
        return nearest

    def robot_model(self, position, robot_size, points, map_glo):
        map_copy = map_glo.copy()
        robot_points = self.range_search(position, robot_size, points)
        for i in range(0, robot_points.shape[0]):
            rob_loc = np.int32(robot_points[i, :])
            rob_loc = np.flipud(rob_loc)
            map_copy[tuple(rob_loc)] = 76
        map_with_robot = map_copy
        return map_with_robot

    def range_search(self, position, r, points):
        nvar = position.shape[0]
        r2 = r ** 2
        s = 0
        for d in range(0, nvar):
            s += (points[:, d] - position[d]) ** 2
        idx = np.nonzero(s <= r2)
        idx = np.asarray(idx).ravel()
        inrange_points = points[idx, :]
        return inrange_points

    def collision_check(self, start_point, end_point, map_size, map_glo):
        x0, y0 = start_point.round()
        x1, y1 = end_point.round()
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        x, y = x0, y0
        error = dx - dy
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        dx *= 2
        dy *= 2

        coll_points = np.ones((1, 2), np.uint8) * -1

        while 0 <= x < map_size[1] and 0 <= y < map_size[0]:
            k = map_glo.item(y, x)
            if k == 1:
                coll_points.itemset((0, 0), x)
                coll_points.itemset((0, 1), y)
                break

            if x == end_point[0] and y == end_point[1]:
                break

            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
        if np.sum(coll_points) == -2:
            coll_index = False
        else:
            coll_index = True

        return coll_points, coll_index

    def inverse_sensor(self, robot_position, sensor_range, op_map, map_glo):
        op_map = inverse_sensor_model(robot_position[0], robot_position[1], sensor_range, op_map, map_glo)
        return op_map

    def frontier(self, op_map, map_size, points):
        y_len = map_size[0]
        x_len = map_size[1]
        mapping = op_map.copy()
        # 0-1 unknown area map
        mapping = (mapping == 127) * 1
        mapping = np.lib.pad(mapping, ((1, 1), (1, 1)), 'constant', constant_values=0)
        fro_map = mapping[2:][:, 1:x_len + 1] + mapping[:y_len][:, 1:x_len + 1] + mapping[1:y_len + 1][:, 2:] + \
                  mapping[1:y_len + 1][:, :x_len] + mapping[:y_len][:, 2:] + mapping[2:][:, :x_len] + mapping[2:][:,
                                                                                                      2:] + \
                  mapping[:y_len][:, :x_len]
        ind_free = np.where(op_map.ravel(order='F') == 255)[0]
        ind_fron_1 = np.where(1 < fro_map.ravel(order='F'))[0]
        ind_fron_2 = np.where(fro_map.ravel(order='F') < 8)[0]
        ind_fron = np.intersect1d(ind_fron_1, ind_fron_2)
        ind_to = np.intersect1d(ind_free, ind_fron)
        f = points[ind_to]
        f = f.astype(int)
        return f[0]

    def unique_rows(self, a):
        a = np.ascontiguousarray(a)
        unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
        result = unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
        result = result[~np.isnan(result).any(axis=1)]
        return result

    def gui(self):
        plt.cla()
        plt.imshow(self.op_map, cmap='gray')
        plt.axis((0, self.map_size[1], self.map_size[0], 0))
        plt.plot(self.xPoint, self.yPoint, 'b', linewidth=2)
        plt.plot(self.robot_position[0], self.robot_position[1], 'mo', markersize=8)
        plt.plot(self.xPoint[0], self.yPoint[0], 'co', markersize=8)
        plt.pause(0.05)
