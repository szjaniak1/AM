#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

#include "parser.hpp"

auto main(int argc, char*argv[]) -> int
{
	const std::string data_directory = "data";
	std::string file_name;
	std::string file_path;
	std::vector<std::vector<weight_type>> graph;
	std::vector<Point> points;
	size_t points_count;

    for (const auto& entry : fs::directory_iterator(data_directory))
    {
        if (entry.is_regular_file())
        {
            file_name = entry.path().filename().stem().string();
            std::cout << file_name << std::endl;

            file_path = "./data/" + file_name + ".tsp";

            auto result_tuple = parse(file_path);
			graph = std::get<0>(result_tuple);
			points = std::get<1>(result_tuple);
			points_count = std::get<2>(result_tuple);

			for (auto p : points)
			{
				std::cout << p.pos_x << " " << p.pos_y << std::endl;
			}
        }
    }

	return EXIT_SUCCESS;
}
