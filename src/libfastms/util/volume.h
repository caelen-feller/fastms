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

#ifndef UTIL_VOL_H
#define UTIL_VOL_H

#include "volume_access.h"
#include "volume_access_convert.h"
#include <iostream>

#ifndef DISABLE_CUDA
#include <cuda_runtime.h>
#endif // not DISABLE_CUDA



// Base class for input images.
// Each input image should know for itself how to convert its data from/to the solver image classes.
class BaseVolume
{
public:
	virtual ~BaseVolume() {}
	virtual BaseVolume* new_of_same_type_and_size() const = 0;
	virtual ArrayDim dim() const = 0;
	virtual void copy_from_layered(const VolumeUntypedAccess<DataInterpretationLayered> &in) = 0;
	virtual void copy_to_layered(VolumeUntypedAccess<DataInterpretationLayered> out) const = 0;
};


template<typename T, typename DataInterpretation>
class VolumeManagerBase
{
public:
	virtual ~VolumeManagerBase() {}
	virtual void copy_from_samekind(VolumeAccess<T, DataInterpretation> out, VolumeAccess<T, DataInterpretation> in) = 0;
	virtual void setzero(VolumeAccess<T, DataInterpretation> image) = 0;
	virtual size_t alloc(VolumeAccess<T, DataInterpretation> &image, const ArrayDim &dim) = 0;
	virtual void free(VolumeAccess<T, DataInterpretation> &image) = 0;
};

template<typename T, typename DataInterpretation, typename Allocator = HostAllocator>
class VolumeManager: public VolumeManagerBase<T, DataInterpretation>
{
public:
	typedef Allocator allocator_t;

	virtual ~VolumeManager() {}

	virtual void copy_from_samekind(VolumeAccess<T, DataInterpretation> out, VolumeAccess<T, DataInterpretation> in)
	{
		allocator_t::copy2d(out.data(), out.data_pitch(), in.const_data(), in.data_pitch(), in.data_width_in_bytes(), in.data_height());
	}

	virtual void setzero(VolumeAccess<T, DataInterpretation> image)
	{
		if (image.is_valid())
		{
			allocator_t::setzero(image.data(), image.num_bytes());
		}
	}

	virtual size_t alloc(VolumeAccess<T, DataInterpretation> &image, const ArrayDim &dim)
	{
		typedef VolumeAccess<T, DataInterpretation> image_access_t;
		bool do_allocation = (!image.is_valid() || image.dim() != dim);
		size_t mem = 0;
		if (do_allocation)
		{
			if (image.is_valid()) { allocator_t::free(image.data()); }
			DataDim data_dim = image_access_t::data_interpretation_t::used_data_dim(dim, sizeof(T));
			void *data = allocator_t::alloc2d(&data_dim);
			image = image_access_t(VolumeData(data, dim, data_dim.pitch), is_on_host());
			allocator_t::setzero(image.data(), image.num_bytes());
			mem = image.num_bytes();
		}
		return mem;
	}

	virtual void free(VolumeAccess<T, DataInterpretation> &image)
	{
		if (image.is_valid()) { allocator_t::free(image.data()); }
	}

private:
	bool is_on_host() { return types_equal<allocator_t, HostAllocator>::value; }
};


// Volume with deep-copy semantics
template<typename T, typename DataInterpretation, typename Allocator = HostAllocator>
class ManagedVolume: public BaseVolume
{
public:
	typedef T elem_t;
	typedef DataInterpretation data_interpretation_t;
	typedef Allocator allocator_t;
	typedef ManagedVolume<elem_t, data_interpretation_t, allocator_t> Self;
	typedef VolumeAccess<elem_t, data_interpretation_t> image_access_t;
	typedef VolumeManager<elem_t, data_interpretation_t, allocator_t> image_manager_t;

	ManagedVolume() : is_owner(true) {}
	ManagedVolume(const ArrayDim &dim) : is_owner(true) { alloc(dim); }
	ManagedVolume(elem_t *data, const ArrayDim &dim) : array(data, dim, is_on_host()), is_owner(false) {}
	ManagedVolume(elem_t *data, const ArrayDim &dim, size_t pitch) : array(data, dim, pitch, ElemType2Kind<T>::value, is_on_host()), is_owner(false) {}
	ManagedVolume(const Self& other) : is_owner(true)
	{
		// copy
		if (!other.array.is_valid()) { return; }
		alloc(other.dim());
		copy_from_samekind(&other);
	}
	virtual ~ManagedVolume() { if (is_owner) { free(); } };
	Self& operator= (const Self &other)
	{
		if (&other != this)
		{
			if (!is_owner)
			{
				is_owner = true;
				array = image_access_t();
			}
			alloc(other.dim());
			copy_from_samekind(&other);
		}
		return *this;
	}
	void copy_from_samekind(const Self *other) { image_manager.copy_from_samekind(this->array, other->array); }

	elem_t* release_data() { is_owner = false; return (elem_t*)array.data(); }
	size_t alloc(const ArrayDim &dim)
	{
		if (!is_owner) { std::cerr << "ERROR: ManagedVolume::alloc(): Currently using external image data, calling alloc() not allowed" << std::endl; return 0; }
		return image_manager.alloc(array, dim);
	}
	void free()
	{
		if (!is_owner) { std::cerr << "ERROR: ManagedVolume::free(): Currently using external image data, calling free() not allowed" << std::endl; return; }
		image_manager.free(array);
	}
	void setzero()
	{
		image_manager.setzero(array);
	}

	typename image_access_t::image_untyped_access_t get_untyped_access() { return array.get_untyped_access(); }
	image_access_t& get_access() { return array; }
	const image_access_t& get_access() const { return array; }
	virtual BaseVolume* new_of_same_type_and_size() const { return new Self(dim()); }
	virtual ArrayDim dim() const { return array.dim(); }
	virtual void copy_from_layered(const VolumeUntypedAccess<DataInterpretationLayered> &in) { copy_image(this->array.get_untyped_access(), in); }
	virtual void copy_to_layered(VolumeUntypedAccess<DataInterpretationLayered> out) const { copy_image(out, this->array.get_untyped_access()); }

private:
	static bool is_on_host() { return types_equal<allocator_t, HostAllocator>::value; }

	image_manager_t image_manager;
	image_access_t array;
	bool is_owner;
};



#endif // UTIL_IMAGE_H
