#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

#include "parser.hpp"

auto main(int argc, char*argv[]) -> int
{
	const std::string data_directory = "data";

    for (const auto& entry : fs::directory_iterator(data_directory)) {
        if (entry.is_regular_file()) {
            std::string file_name = entry.path().filename().stem().string();
            std::cout << file_name << std::endl;

            std::string file_path = "./data/" + file_name + ".tsp";
            auto result_tuple = parse(file_path);
			std::vector<std::vector<weight_type>> graph = std::get<0>(result_tuple);
			std::vector<Point> points = std::get<1>(result_tuple);
			size_t points_count = std::get<2>(result_tuple);

			for (auto p : points)
			{
				std::cout << p.pos_x << " " << p.pos_y << std::endl;
			}
        }
    }

	return EXIT_SUCCESS;
}
