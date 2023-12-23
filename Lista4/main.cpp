#include <iostream>
#include <string>

#include "parser.hpp"

auto main(int argc, char*argv[]) -> int
{
	std::string file_path = "./data/xqf131.tsp";

	auto result_tuple = parse(file_path);
    std::vector<std::vector<weight_type>> graph = std::get<0>(result_tuple);
    std::vector<Point> points = std::get<1>(result_tuple);
    size_t points_count = std::get<2>(result_tuple);

    for (auto p : points)
    {
    	std::cout << p.pos_x << " " << p.pos_y << std::endl;
    }

	return EXIT_SUCCESS;
}
