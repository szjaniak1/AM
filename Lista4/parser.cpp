#include "parser.hpp"

#include <fstream>
#include <sstream>
#include <cmath>

auto calculate_weight(const std::vector<Point>& points) -> std::tuple<std::vector<std::vector<weight_type>>, std::vector<Point>, const size_t>;
auto get_weight(const Point& p1, const Point& p2) -> weight_type;

auto parse(const std::string file_path) -> std::tuple<std::vector<std::vector<weight_type>>, std::vector<Point>, const size_t>
{
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        // Handle file opening error...
    }

    std::string dim, dim_num;
    std::getline(file, dim, ':');
    std::getline(file, dim_num);
    const size_t size = std::stoi(dim_num);

    std::vector<Point> points(size, {0, 0});

    std::string line;
    size_t num;
    uint16_t x, y;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        iss >> num >> x >> y;
        points[num - 1] = {x, y};
    }

    return calculate_weight(points);
}

auto calculate_weight(const std::vector<Point>& points) -> std::tuple<std::vector<std::vector<weight_type>>, std::vector<Point>, const size_t>
{
	const size_t size = points.size();
    std::vector<std::vector<weight_type>> result(size, std::vector<weight_type>(size, 0));

    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < size; ++j)
        {
            result[i][j] = get_weight(points[i], points[j]);
        }
    }

    return std::make_tuple(result, points, size);
}

auto get_weight(const Point& point1, const Point& point2) -> weight_type
{
	return static_cast<weight_type>(std::round(std::sqrt(std::pow((point1.pos_x - point2.pos_x), 2) + std::pow((point1.pos_y - point2.pos_y), 2))));
}
