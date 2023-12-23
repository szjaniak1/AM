#ifndef PARSER_HPP
#define PARSER_HPP

#include <string>
#include <vector>
#include <cstdint>

using weight_type = uint32_t;

struct Point
{
	uint16_t pos_x;
	uint16_t pos_y;
};

auto parse(const std::string file_path) -> std::tuple<std::vector<std::vector<weight_type>>, std::vector<Point>, const size_t>;

#endif /* PARSER_HPP */
