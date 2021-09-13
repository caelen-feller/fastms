
/*
* This file is part of fastms.
*
* Copyright 2014 Evgeny Strekalovskiy <evgeny dot strekalovskiy at in dot tum dot de> (Technical University of Munich)
*
* fastms is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* fastms is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with fastms. If not, see <http://www.gnu.org/licenses/>.
*/


#include "volume_mat.h"
#include <iostream>


// cv::Mat extract_row(const cv::Mat in_mat, int row)
// {
// 	// row image will have the same number of channels as the input image
// 	int w = in_mat.cols;
// 	int h = in_mat.rows;
// 	if (row < 0 || row >= h)
// 	{
// 		int row_new = std::max(0, std::min(h - 1, row));
// 		std::cerr << "WARNING: extract_row: " << row << " is not a valid row (0 .. " << h - 1 << "), using row = " << row_new << std::endl;
// 		row = row_new;
// 	}
// 	cv::Mat mat_row(1, w, in_mat.type(), cv::Scalar::all(0));
// 	memcpy(mat_row.data, in_mat.ptr(row), (size_t)w * in_mat.elemSize());
// 	return mat_row;
// }
