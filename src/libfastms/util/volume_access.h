/*
* This file is an extension of fastms.
*
* Copyright 2021 Caelen Feller <fellerc@tcd.ie> (Trinity College Dublin)
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

#ifndef UTIL_VOL_ACCESS_H
#define UTIL_VOL_ACCESS_H

#include <cstring>  // for memset, memcpy
#include "real.h"
#include "types_equal.h"
#include <ostream>

#ifndef DISABLE_CUDA
#include <cuda_runtime.h>
#endif // not DISABLE_CUDA




struct DataIndex3
{
	HOST_DEVICE DataIndex3() : x(0), y(0), z(0) {}
	HOST_DEVICE DataIndex3(size_t x, size_t y, size_t z) : x(x), y(y), z(z) {}
	size_t x;
	size_t y;
	size_t z;
};


// Pitch Explained https://stackoverflow.com/a/16119944
struct DataDim3
{
	HOST_DEVICE DataDim3() : pitch(0), height(0), depth(0) {}
	HOST_DEVICE DataDim3(size_t pitch, size_t height, size_t depth) : pitch(pitch), height(height), depth(depth) {}
	HOST_DEVICE size_t num_bytes() const { return pitch * height * depth; }
	size_t pitch;
	size_t height;
	size_t depth;
};


struct Dim3D
{
	HOST_DEVICE Dim3D() : w(0), h(0), d(0) {}
	HOST_DEVICE Dim3D(int w, int h, int d) : w(w), h(h), d(d) {}
	HOST_DEVICE size_t num_elem() const { return (size_t)w * h * d; }
	HOST_DEVICE bool operator== (const Dim3D &other_dim) { return w == other_dim.w && h == other_dim.h && d == other_dim.d; }
	HOST_DEVICE bool operator!= (const Dim3D &other_dim) { return !(operator==(other_dim)); }

	int w;
	int h;
	int d;
};


struct ArrayDim3
{
	HOST_DEVICE ArrayDim3() : w(0), h(0), d(0), num_channels(0) {}
	HOST_DEVICE ArrayDim3(int w, int h, int d, int num_channels) : w(w), h(h), d(d), num_channels(num_channels) {}
	HOST_DEVICE Dim3D dim3d() const { return Dim3D(w, h, d); }
	HOST_DEVICE size_t num_elem() const { return (size_t)w * h * d * num_channels; }
	HOST_DEVICE bool operator== (const ArrayDim3 &other_dim) { return w == other_dim.w && h == other_dim.h && d == other_dim.d && num_channels == other_dim.num_channels; }
	HOST_DEVICE bool operator!= (const ArrayDim3 &other_dim) { return !(operator==(other_dim)); }

	int w;
	int h;
	int d;
	int num_channels;
};
inline std::ostream& operator<< (std::ostream &out, const ArrayDim3 &dim)
{
	out << dim.w << " x " << dim.h << " x " << dim.num_channels;
	return out;
}


class HostAllocator3
{
public:
	static void free(void *&ptr) { delete[] (char*)ptr; ptr = NULL; }
	static void setzero(void *ptr, size_t num_bytes) { memset(ptr, 0, num_bytes); }
	static void* alloc3d(DataDim3 *used_data_dim)
	{
		return (void*)(new char[used_data_dim->num_bytes()]);
	}
	static void copy3d(void *out, size_t out_pitch, const void *in, size_t in_pitch, size_t w_bytes, size_t h_lines, size_t d_slices)
	{
		if (out_pitch == in_pitch)
		{
			size_t num_bytes = in_pitch * h_lines * d_slices;
			memcpy(out, in, num_bytes);
		}
		else
		{
			for (size_t y = 0; y < h_lines; y++)
			{
				for (size_t z = 0; z < d_slices; z++)
				{
					const void *in_slice = (const void*)((const char*)in + in_pitch * y + in_pitch * h_lines * z);
					void *out_slice = (void*)((char*)out + out_pitch * y + out_pitch * h_lines * z);
					memcpy(out_slice, in_slice, w_bytes);
				}
			}
		}
	}
};


#ifndef DISABLE_CUDA
class DeviceAllocator3
{
public:
	static void free(void *&ptr_cuda) { cudaFree(ptr_cuda); ptr_cuda = NULL; }
	static void setzero(void *ptr_cuda, size_t num_bytes)
	{
		cudaMemset(ptr_cuda, 0, num_bytes);
	}
	static void* alloc3d(DataDim3 *used_data_dim)
	{
		void *ptr_cuda = NULL;
		size_t pitch0 = 0;

		cudaMallocPitch(&ptr_cuda, &pitch0, used_data_dim->pitch, used_data_dim->height * used_data_dim->depth);
		
		used_data_dim->pitch = pitch0;
		return ptr_cuda;
	}
	static void copy3d(void *out_cuda, size_t out_pitch, const void *in_cuda, size_t in_pitch, size_t w_bytes, size_t h_lines, size_t d_slices)
	{
		cudaMemcpy2D(out_cuda, out_pitch, in_cuda, in_pitch, w_bytes, h_lines*d_slices, cudaMemcpyDeviceToDevice);
	}
	static void copy3d_h2d(void *out_cuda, size_t out_pitch, const void *in_host, size_t in_pitch, size_t w_bytes, size_t h_lines, size_t d_slices)
	{
		cudaMemcpy2D(out_cuda, out_pitch, in_host, in_pitch, w_bytes, h_lines*d_slices, cudaMemcpyHostToDevice);
	}
	static void copy3d_d2h(void *out_host, size_t out_pitch, const void *in_cuda, size_t in_pitch, size_t w_bytes, size_t h_lines, size_t d_slices)
	{
		cudaMemcpy2D(out_host, out_pitch, in_cuda, in_pitch, w_bytes, h_lines*d_slices, cudaMemcpyDeviceToHost);
	}
};
#endif // not DISABLE_CUDA


struct VolumeData
{
	HOST_DEVICE VolumeData() : data_(NULL), data_pitch_(0) {}
	HOST_DEVICE VolumeData(void *data, const ArrayDim3 &dim, size_t data_pitch) : data_(data), dim_(dim), data_pitch_(data_pitch) {}
	void *data_;
	ArrayDim3 dim_;
	size_t data_pitch_;
};


template<typename TUntypedAccess>
TUntypedAccess alloc_untyped_access(const ArrayDim3 &dim, ElemKind elem_kind, bool on_host)
{
	DataDim3 data_dim = TUntypedAccess::data_interpretation_t::used_data_dim(dim, ElemKindGeneral::size(elem_kind));
#ifndef DISABLE_CUDA
	void *data = (on_host ? HostAllocator3::alloc3d(&data_dim) : DeviceAllocator3::alloc3d(&data_dim));
#else
	void *data = HostAllocator3::alloc3d(&data_dim);
#endif // not DISABLE_CUDA
	return TUntypedAccess(VolumeData(data, dim, data_dim.pitch), elem_kind, on_host);
}


template<typename DataInterpretation> struct VolumeUntypedAccess;

template<typename T, typename DataInterpretation>
struct VolumeAccess
{
	typedef T elem_t;
	typedef DataInterpretation data_interpretation_t;
	typedef VolumeUntypedAccess<data_interpretation_t> volume_untyped_access_t;

	HOST_DEVICE VolumeAccess() : is_on_host_(true) {}
	HOST_DEVICE VolumeAccess(void *data, const ArrayDim3 &dim, bool is_on_host) :
			volume_data_(data, dim, data_interpretation_t::used_data_dim(dim, sizeof(T)).pitch), is_on_host_(is_on_host) {}
	HOST_DEVICE VolumeAccess(const VolumeData &volume_data, bool is_on_host) :
			volume_data_(volume_data), is_on_host_(is_on_host) {}

	HOST_DEVICE volume_untyped_access_t get_untyped_access() const { return volume_untyped_access_t(volume_data_, ElemType2Kind<T>::value, is_on_host_); }

	HOST_DEVICE T& get(int x, int y, int z, int i) { return *(T*)((char*)volume_data_.data_ + offset(x, y, z, i)); }
	HOST_DEVICE const T& get(int x, int y, int z, int i) const { return *(T*)((char*)volume_data_.data_ + offset(x, y, z, i)); }
	HOST_DEVICE ArrayDim3 dim() const { return volume_data_.dim_; }
	HOST_DEVICE bool is_valid() const { return volume_data_.data_ != NULL; }
	HOST_DEVICE bool is_on_host() const { return is_on_host_; }

	HOST_DEVICE void*& data() { return volume_data_.data_; }
	HOST_DEVICE const void* const_data() const { return volume_data_.data_; }
	HOST_DEVICE size_t data_pitch() const { return volume_data_.data_pitch_; }
	HOST_DEVICE size_t data_height() const { return used_data_dim().height; }
	HOST_DEVICE size_t data_depth() const { return used_data_dim().depth; }
	HOST_DEVICE size_t data_width_in_bytes() const { return used_data_dim().pitch; }
	HOST_DEVICE size_t num_bytes() const { return data_pitch() * data_height() * data_depth(); }

private:
	HOST_DEVICE size_t offset(int x, int y, int z, int i) const
	{
		const DataIndex3 &data_index = data_interpretation_t::get(x, y, z, i, volume_data_.dim_);
		// This might not be right! 
		return data_index.x * sizeof(T) + volume_data_.data_pitch_ * data_index.y  +  volume_data_.data_pitch_ * volume_data_.dim_.h * data_index.z);
	}
	HOST_DEVICE DataDim3 used_data_dim() const { return data_interpretation_t::used_data_dim(volume_data_.dim_, sizeof(T)); }

	VolumeData volume_data_;
	bool is_on_host_;
};


template<typename DataInterpretation>
struct VolumeUntypedAccess
{
	typedef DataInterpretation data_interpretation_t;
	template<typename T> struct volume_access_t { typedef VolumeAccess<T, data_interpretation_t> type; };

	HOST_DEVICE VolumeUntypedAccess() : elem_kind_(elem_kind_uchar), is_on_host_(true) {}
	HOST_DEVICE VolumeUntypedAccess(void *data, const ArrayDim3 &dim, ElemKind elem_kind, bool is_on_host) :
			volume_data_(data, dim, data_interpretation_t::used_data_dim(dim, ElemKindGeneral::size(elem_kind)).pitch),
			elem_kind_(elem_kind), is_on_host_(is_on_host) {}
	HOST_DEVICE VolumeUntypedAccess(const VolumeData &volume_data, ElemKind elem_kind, bool is_on_host) :
			volume_data_(volume_data),
			elem_kind_(elem_kind), is_on_host_(is_on_host) {}

	template<typename T> HOST_DEVICE typename volume_access_t<T>::type get_access() const { return typename volume_access_t<T>::type(volume_data_, is_on_host_); }
	HOST_DEVICE ElemKind elem_kind() const { return elem_kind_; }

	HOST_DEVICE void* get_address(int x, int y, int z, int i) { return (void*)((char*)volume_data_.data_ + offset_address(x, y, z, i)); }
	HOST_DEVICE const void* get_address(int x, int y, int z, int i) const { return (const void*)((const char*)volume_data_.data_ + offset_address(x, y, z, i)); }
	HOST_DEVICE ArrayDim3 dim() const { return volume_data_.dim_; }
	HOST_DEVICE bool is_valid() const { return volume_data_.data_ != NULL; }
	HOST_DEVICE bool is_on_host() const { return is_on_host_; }

	HOST_DEVICE void*& data() { return volume_data_.data_; }
	HOST_DEVICE const void* const_data() const { return volume_data_.data_; }
	HOST_DEVICE size_t data_pitch() const { return volume_data_.data_pitch_; }
	HOST_DEVICE size_t data_height() const { return used_data_dim().height; }
	HOST_DEVICE size_t data_width_in_bytes() const { return used_data_dim().pitch; }
	HOST_DEVICE size_t num_bytes() const { return data_pitch() * data_height(); }

private:
	HOST_DEVICE size_t offset_address(int x, int y, int z, int i) const
	{
		const DataIndex3 &data_index = data_interpretation_t::get(x, y, z, i, volume_data_.dim_);
		// THis might not be right!!!
		return data_index.x * elem_size() + volume_data_.data_pitch_ * data_index.y + volume_data_.data_pitch_ * volume_data_.dim_.h * data_index.z;
	}
	HOST_DEVICE DataDim3 used_data_dim() const { return data_interpretation_t::used_data_dim(volume_data_.dim_, elem_size()); }
	HOST_DEVICE size_t elem_size() const { return ElemKindGeneral::size(elem_kind_); }

	VolumeData volume_data_;
	ElemKind elem_kind_;
	bool is_on_host_;
};


struct DataInterpretationLayered
{
	HOST_DEVICE static DataIndex3 get(int x, int y, int z, int i, const ArrayDim3 &dim)
	{
		return DataIndex3(x, y, z + (size_t)dim.z * i);
	}
	HOST_DEVICE static DataDim3 used_data_dim (const ArrayDim3 &dim, size_t elem_size)
	{
		return DataDim3(dim.w * elem_size, (size_t) dim.h, (size_t) dim.d * dim.num_channels);
	}
};


struct DataInterpretationLayeredTransposed
{
	HOST_DEVICE static DataIndex3 get(int x, int y, int z, int i, const ArrayDim3 &dim)
	{
		return DataIndex3(z, y, x + (size_t)dim.w * i);
	}

	HOST_DEVICE static DataDim3 used_data_dim (const ArrayDim3 &dim, size_t elem_size)
	{
		return DataDim3(dim.d * elem_size, (size_t) dim.h, (size_t)dim.w * dim.num_channels);
	}
};


// TODO
struct DataInterpretationInterlaced
{
	HOST_DEVICE static DataIndex3 get(int x, int y, int z, int i, const ArrayDim3 &dim)
	{
		return DataIndex3(i + (size_t)dim.num_channels * x, y, z);
	}
	HOST_DEVICE static DataDim3 used_data_dim (const ArrayDim3 &dim, size_t elem_size)
	{
		return DataDim3((size_t)dim.num_channels * dim.w * elem_size, dim.h, dim.d);
	}
};


struct DataInterpretationInterlacedReversed
{
	HOST_DEVICE static DataIndex3 get(int x, int y, int z, int i, const ArrayDim3 &dim)
	{
		return DataIndex3((dim.num_channels - 1 - i) + (size_t)dim.num_channels * x, y, z);
	}
	HOST_DEVICE static DataDim3 used_data_dim (const ArrayDim3 &dim, size_t elem_size)
	{
		return DataDim3((size_t)dim.num_channels * dim.w * elem_size, dim.h, dim.d);
	}
};



#undef HOST_DEVICE
#undef FORCEINLINE

#endif // UTIL_IMAGE_ACCESS_H
