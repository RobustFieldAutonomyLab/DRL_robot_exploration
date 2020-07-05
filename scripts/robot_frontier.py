import numpy as np
import cv2
from scipy import spatial
# from numba import jit
from sklearn.neighbors import NearestNeighbors
import time
from scipy import ndimage
import matplotlib.pyplot as plt


class Robot:
    def __init__(self):
        self.global_map = map_setup('map.png')
        self.op_map = np.ones(self.global_map.shape) * 127
        self.map_size = np.shape(self.global_map)
        self.resolution = 1
        self.sensor_range = 80
        self.robot_step = np.array([self.sensor_range / 2, self.sensor_range])
        self.robot_position = np.array([49, 360])
        self.old_position = np.zeros([2])
        self.old_op_map = np.empty([0])
        self.direction = all_direction(8)
        self.t = map_points(self.global_map)
        self.free_tree = spatial.KDTree(free_points(self.global_map).tolist())
        self.robot_size = 6
        self.trap = 0

    def begin(self):
        inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)
        step_map = robot_model(self.robot_position, self.robot_size, self.t, self.op_map)
        return step_map

    def step(self, action_index):
        # plt.ion()
        terminal = False
        complete = False
        new_location = False
        self.old_position = self.robot_position.copy()
        self.old_op_map = self.op_map.copy()
        # take action
        take_action(action_index, self.direction, self.robot_position, self.robot_step)
        # collision check
        collision_points, collision_index = collision_check(self.old_position, self.robot_position, self.map_size,
                                                            self.global_map)
        if collision_index:
            terminal = True
            self.robot_position = nearest_free(self.free_tree, collision_points)
            inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)
            step_map = robot_model(self.robot_position, self.robot_size, self.t, self.op_map)
        else:
            inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)
            step_map = robot_model(self.robot_position, self.robot_size, self.t, self.op_map)

        reward = get_reward(self.old_op_map, self.op_map, collision_index)

        if reward == 0 or reward == -1:
            self.trap += 1
            if self.trap == 3:
                new_location = True
        else:
            self.trap = 0

        if terminal:
            self.robot_position = self.old_position.copy()
            self.op_map = self.old_op_map.copy()

        if np.size(np.where(self.global_map == 255)) - np.size(np.where(self.op_map == 255)) < 500:
            self.__init__()
            complete = True
        # plt.imshow(step_map, cmap='gray')
        # plt.draw()
        # plt.show()
        return step_map, reward, terminal, complete, new_location, collision_index

    def rescuer(self):
        # plt.ion()
        ss = time.time()
        self.robot_position = frontier(self.op_map, self.map_size)
        ee = time.time()
        # print ("points", self.robot_position, "time", ee-ss)
        inverse_sensor(self.robot_position, self.sensor_range, self.op_map, self.global_map)
        step_map = robot_model(self.robot_position, self.robot_size, self.t, self.op_map)
        self.trap = 0
        # plt.imshow(step_map, cmap='gray')
        # plt.draw()
        # plt.show()
        return step_map


def all_direction(number):
    directions = np.zeros([0])
    for i in range(0, number):
        directions = np.append(directions, 2 * np.pi / number * i)
    return directions


def take_action(action_index, all_dir, robot_position, step_length):
    if action_index < 8:
        angle = all_dir[action_index]
        x = step_length[0] * np.cos(-angle)
        y = step_length[0] * np.sin(-angle)
    else:
        angle = all_dir[action_index-8]
        x = step_length[1] * np.cos(-angle)
        y = step_length[1] * np.sin(-angle)
    robot_position[0] = np.round(robot_position[0] + x)
    robot_position[1] = np.round(robot_position[1] + y)


def map_setup(location):
    global_map = cv2.imread(location, 0)
    global_map = (global_map > 200)
    global_map = global_map * 254 + 1
    return global_map


def map_points(map_glo):
    map_x = map_glo.shape[1]
    map_y = map_glo.shape[0]
    x = np.linspace(0, map_x - 1, map_x)
    y = np.linspace(0, map_y - 1, map_y)
    t1, t2 = np.meshgrid(x, y)
    points = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
    return points


def free_points(op_map):
    index = np.where(op_map == 255)
    free = np.asarray([index[1], index[0]]).T
    return free


def get_reward(old_op_map, op_map, coll_index):
    if not coll_index:
        reward = float(np.size(np.where(op_map == 255)) - np.size(np.where(old_op_map == 255)))/14000
        if reward > 1:
            reward = 1
    else:
        reward = -1
    return reward


def nearest_free(tree, point):
    pts = np.atleast_2d(point)
    index = tuple(tree.query(pts)[1])
    nearest = tree.data[index]
    return nearest


def robot_model(position, robot_size, points, map_glo):
    map_copy = map_glo.copy()
    robot_points = range_search(position, robot_size, points)
    for i in range(0, robot_points.shape[0]):
        rob_loc = np.int32(robot_points[i, :])
        rob_loc = np.flipud(rob_loc)
        map_copy[tuple(rob_loc)] = 76
    map_with_robot = map_copy
    return map_with_robot


def range_search(position, r, points):
    nvar = position.shape[0]
    r2 = r ** 2
    s = 0
    for d in range(0, nvar):
        s += (points[:, d] - position[d]) ** 2
    idx = np.nonzero(s <= r2)
    idx = np.asarray(idx).ravel()
    inrange_points = points[idx, :]
    return inrange_points


def castray(start_point, end_point, map_size, map_glo):
    x0, y0 = start_point.round()
    x1, y1 = end_point.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    coll_flag = 0
    coll_size = 5

    while 0 <= x < map_size[1] and 0 <= y < map_size[0]:
        if x == x1 and y == y1:
            break

        if map_glo[y][x] != 1 and coll_flag > 0:
            break
        elif map_glo[y][x] == 1 and coll_flag < coll_size:
            coll_flag += 1

        yield y, x
        if map_glo[y][x] == 1 and coll_flag == coll_size:
            break

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx


def collision_check(start_point, end_point, map_size, map_glo):
    x0, y0 = start_point.round()
    x1, y1 = end_point.round()
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    x, y = x0, y0
    error = dx - dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    dx *= 2
    dy *= 2

    coll_points = np.empty([0])

    while 0 <= x < map_size[1] and 0 <= y < map_size[0]:
        if map_glo[y][x] == 1:
            coll_points = np.array([x, y])
            break

        if x == end_point[0] and y == end_point[1]:
            break

        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    if np.size(coll_points) == 0:
        coll_index = False
    else:
        coll_index = True

    return coll_points, coll_index


def inverse_sensor(robot_position, sensor_range, op_map, map_glo):
    map_shape = np.shape(op_map)
    sensor_angle_inc = np.deg2rad(0.7)
    sensor_angle_range = 2 * np.pi

    start_angle = 0
    end_angle = start_angle + sensor_angle_range

    for angle in np.arange(start_angle, end_angle, sensor_angle_inc):
        ray_end = (robot_position + sensor_range * np.array([np.cos(angle), np.sin(angle)]))

        for points in castray(robot_position, ray_end, map_shape, map_glo):
            op_map[points] = map_glo[points]


def frontier(op_map, map_size):
    free_flag = 0
    occ_flag = 0
    unknown_flag = 0
    for y in range(map_size[0]):
        for x in range(map_size[1]):
            if not op_map[y][x] == 255:
                continue
            else:
                try:
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            if op_map[y + i][x + j] == 255:
                                free_flag += 1
                            elif op_map[y + i][x + j] == 1:
                                occ_flag += 1
                            elif op_map[y + i][x + j] == 127:
                                unknown_flag += 1
                except IndexError:
                    continue
                if unknown_flag > 0 and free_flag > 1 and occ_flag == 0:
                    f = np.array([x, y])
                    return f
                elif unknown_flag > 4:
                    f = np.array([x, y])
                    return f
                free_flag = 0
                occ_flag = 0
                unknown_flag = 0


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)] * a.shape[1]))
    result = unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
    result = result[~np.isnan(result).any(axis=1)]
    return result
