#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <Eigen/Dense>
#include <cmath>

namespace py = pybind11;

void line(double fx1, double fy1, double fx2, double fy2, const Eigen::MatrixXd& global_map,
          Eigen::MatrixXd& op_map) {
  int x1 = static_cast<int>(round(fx1));
  int y1 = static_cast<int>(round(fy1));
  int x2 = static_cast<int>(round(fx2));
  int y2 = static_cast<int>(round(fy2));

  int dx = abs(x2 - x1);
  int dy = abs(y2 - y1);
  int x = x1, y = y1;
  int error = dx - dy;
  int x_inc = x2 > x1 ? 1 : -1;
  int y_inc = y2 > y1 ? 1 : -1;
  dx *= 2;
  dy *= 2;

  int coll_flag = 0;
  int coll_size = 10;

  int rows = global_map.rows();
  int cols = global_map.cols();

  while (x >= 0 && x < cols && y >= 0 && y < rows) {
    if (x == x2 && y == y2) break;

    int k = global_map(y, x);
    if (k != 1 && coll_flag > 0) break;
    if (k == 1 && coll_flag < coll_size) coll_flag += 1;

    op_map(y, x) = k;

    if (k == 1 && coll_flag == coll_size) break;

    if (error > 0) {
      x += x_inc;
      error -= dy;
    } else {
      y += y_inc;
      error += dx;
    }
  }
}

Eigen::MatrixXd inverse_sensor_model(int x0, int y0, int sensor_range,
                            Eigen::MatrixXd op_map, Eigen::MatrixXd global_map) {
  Eigen::MatrixXd op_map_mod(op_map);
  double sensor_angle_inc = 0.5 / 180.0 * M_PI;
  for (double angle = 0.0; angle < M_PI * 2.0; angle += sensor_angle_inc) {
    double x1 = double(x0) + double(sensor_range) * cos(angle);
    double y1 = double (y0) + double(sensor_range) * sin(angle);
    line(x0, y0, x1, y1, global_map, op_map_mod);
  }
  return op_map_mod;
}

PYBIND11_PLUGIN(inverse_sensor_model) {
  py::module m("inverse_sensor_model", "inverse_sensor_model");
  m.def("inverse_sensor_model", &inverse_sensor_model, "inverse_sensor_model");
  return m.ptr();
}
