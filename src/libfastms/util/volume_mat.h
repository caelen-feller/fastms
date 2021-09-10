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

#ifndef UTIL_VOL_MAT_H
#define UTIL_VOL_MAT_H

#include "volume.h"
#include <vector>

struct VolMat{
	VolMat(const ArrayDim3 dim, int depth) : dim(dim), depth(depth) { data = std::vector<unsigned char>(dim.num_elem());}
	std::vector<unsigned char> data;
	ArrayDim3 dim;
	int depth;
}


void show_volume(std::string title, const VolMat vol, int x_window, int y_window, int z_window);
// float* image1d_to_graph(const cv::Mat &mat, int graph_height, int channel_id = -1, double thresh_jump = -1.0);
VolMat extract_slice(const VolMat vol, int slice);  // extract a slice from a 3d volume, to process this slice as a single 2d image

// TODO: Dynamic bit depth
struct VolDepth { static const int value =  256; };

class MatVolume: public BaseVolume
{
public:
	MatVolume(VolMat volume) { mat = volume; }
	MatVolume(const ArrayDim3 &dim, int depth) { mat = VolMat(dim, depth); }
	virtual ~MatVolume() {}

	virtual BaseVolume* new_of_same_type_and_size() const { return new MatVolume(dim(), mat.depth()); }
	virtual ArrayDim3 dim() const { return ArrayDim3(mat.dim.w, mat.dim.h, mat.dim.d, mat.dim.num_channels); }
	virtual void copy_from_layered(const VolumeUntypedAccess<DataInterpretationLayered> &in) { copy_volume(this->get_untyped_access(), in); }
	virtual void copy_to_layered(VolumeUntypedAccess<DataInterpretationLayered> out) const { copy_volume(out, this->get_untyped_access()); }

	VolMat get_mat() const { return mat; }

private:
	typedef VolumeUntypedAccess<DataInterpretationInterlacedReversed> volume_untyped_access_t;
	volume_untyped_access_t get_untyped_access() const
	{
		return volume_untyped_access_t(get_data(), dim(), elem_kind(), true);  // true = on_host
	}

	void* get_data() const { return (void*)mat.data; }
	ElemKind elem_kind() const { return elem_kind_uchar; } // TODO: Dynamic typing

	VolMat mat;
};


#endif // UTIL_VOL_MAT_H