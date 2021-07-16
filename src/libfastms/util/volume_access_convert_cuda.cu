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

#if !defined(DISABLE_CUDA) && defined(__CUDACC__)

#include "volume_access_convert.h"
#include "volume_vars_cuda.cuh"
#include "check_cuda.cuh"



template<typename TUntypedAccessOut, typename TUntypedAccessIn>
__global__ void copy_volume_d2d_base_kernel (TUntypedAccessOut out, TUntypedAccessIn in)
{
	const ElemKind out_kind = out.elem_kind();
	const ElemKind in_kind = in.elem_kind();
	const Dim3D &dim3d = in.dim().dim3d();
	const int num_channels = in.dim().num_channels;
	int x = cuda_x();
	int y = cuda_y();
	int z = cuda_z();
	if (is_active(x, y, z, dim3d))
	{
		for (int i = 0; i < num_channels; i++)
		{
			convert_type(out_kind, in_kind, out.get_address(x, y, z, i), in.get_address(x, y, z, i));
		}
	}
}


template<typename TUntypedAccessOut, typename TUntypedAccessIn>
void copy_volume_d2d_base(TUntypedAccessOut out, TUntypedAccessIn in)
{
	const Dim3D &dim3d = in.dim().dim3d();
	dim3 block = cuda_block_size(dim3d.w, dim3d.h, dim3d.d);
	dim3 grid = cuda_grid_size(block, dim3d.w, dim3d.h, dim3d.d);
	copy_volume_d2d_base_kernel <<<grid, block>>> (out, in);  CUDA_CHECK;
}


#define COPY_Iout_Iin(Iout, Iin) template void copy_volume_d2d_base<VolumeUntypedAccess<Iout>, VolumeUntypedAccess<Iin> >(VolumeUntypedAccess<Iout>, VolumeUntypedAccess<Iin>);
COPY_Iout_Iin(DataInterpretationLayered, DataInterpretationLayered)

COPY_Iout_Iin(DataInterpretationLayeredTransposed, DataInterpretationLayered)
COPY_Iout_Iin(DataInterpretationLayered, DataInterpretationLayeredTransposed)

COPY_Iout_Iin(DataInterpretationInterlaced, DataInterpretationLayered)
COPY_Iout_Iin(DataInterpretationLayered, DataInterpretationInterlaced)

COPY_Iout_Iin(DataInterpretationInterlacedReversed, DataInterpretationLayered)
COPY_Iout_Iin(DataInterpretationLayered, DataInterpretationInterlacedReversed)
#undef COPY_Iout_Iin



#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)
