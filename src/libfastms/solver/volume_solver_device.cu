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

#include "volume_solver_device.h"
#include "volume_solver_base.h"
#include "util/mem_cuda.cuh"
#include "util/sum_cuda.cuh"
#include "util/timer_cuda.cuh"
#include "util/check_cuda.cuh"
#include "util/volume_vars_cuda.cuh"



template<typename real>
class DeviceEngine3: public Engine3<real>
{
public:
	typedef Engine3<real> Base;
	typedef typename Base::volume_access_t volume_access_t;
	typedef typename Base::linear_operator_t linear_operator_t;
	typedef typename Base::regularizer_t regularizer_t;
	typedef typename Base::dataterm_t dataterm_t;
	typedef DeviceAllocator3 allocator_t;
	typedef VolumeManager<real, typename volume_access_t::data_interpretation_t, allocator_t> volume_manager_t;

	DeviceEngine3()
	{
		summator.alloc();
		is_enabled = gpu_supports_real<real>(); if (!is_enabled) { std::cerr << "ERROR: VolumeSolverDevice<double>: Current GPU does not support double. Function calls will have no effect" << std::endl; }
	}
	virtual ~DeviceEngine3()
	{
		summator.free();
	}
	virtual std::string str() { return "cuda"; }
	virtual void alloc(const ArrayDim3 &dim_u)
	{
		block = cuda_block_size(dim_u.w, dim_u.h, dim_u.d);
		grid = cuda_grid_size(block, dim_u.w, dim_u.h, dim_u.d);
	}
	virtual void free()	{}
	virtual bool is_valid() { return is_enabled; }
	virtual typename Base::volume_manager_base_t* volume_manager() { return &volume_manager_; }
	virtual real get_sum(volume_access_t a) { return summator.sum(a.const_data(), a.num_bytes()); }
	virtual void timer_start() { timer.start(); }
	virtual void timer_end() { timer.end(); }
	virtual double timer_get() { return timer.get(); }
	virtual void synchronize() { cudaDeviceSynchronize(); CUDA_CHECK; }

	virtual void run_dual_p(volume_access_t p, volume_access_t u, linear_operator_t linear_operator, regularizer_t regularizer, real dt);
	virtual void run_prim_u(volume_access_t u, volume_access_t ubar, volume_access_t p, linear_operator_t linear_operator, dataterm_t dataterm, real theta_bar, real dt);
	virtual void energy_base(volume_access_t u, volume_access_t aux_reduce, linear_operator_t linear_operator, dataterm_t dataterm, regularizer_t regularizer);
	virtual void add_edges(volume_access_t cur_result, linear_operator_t linear_operator, regularizer_t regularizer);
	virtual void set_regularizer_weight_from__normgrad(volume_access_t regularizer_weight, volume_access_t volume, linear_operator_t linear_operator);
	virtual void set_regularizer_weight_from__exp(volume_access_t regularizer_weight, real coeff);
	virtual void diff_l1_base(volume_access_t a, volume_access_t b, volume_access_t aux_reduce);

	volume_manager_t volume_manager_;
    DeviceTimer timer;
    DeviceSummator<real> summator;
	dim3 grid;
	dim3 block;
	bool is_enabled;
};


template<typename TVolumeAccess, typename TLinearOperator, typename TRegularizer>
__device__ void cuda_run_dual_p_device (const int u_num_channels, TVolumeAccess p, TVolumeAccess u, TLinearOperator linear_operator, TRegularizer regularizer, typename TVolumeAccess::elem_t dt)
{
	typedef typename TVolumeAccess::elem_t real;

	const Dim3D &dim3d = p.dim().dim3d();
    const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
	int x = cuda_x();
	int y = cuda_y();
	int z = cuda_z();
	if (is_active(x, y, z, dim3d))
	{
		ShMemArray<real> p_sh(p_num_channels);

		linear_operator.apply(p_sh, u, x, y, z, dim3d, u_num_channels);

		for(int i = 0; i < p_num_channels; i++)
		{
			p_sh.get(i) = p.get(x, y, z, i) + p_sh.get(i) * dt;
		}

		regularizer.prox_star(p_sh, dt, x, y, z, dim3d, p_num_channels);

		for(int i = 0; i < p_num_channels; i++)
		{
			p.get(x, y, z, i) = p_sh.get(i);
		}
	}
}
template<typename TVolumeAccess, typename TLinearOperator, typename TRegularizer>
__global__ void cuda_run_dual_p_kernel (TVolumeAccess p, TVolumeAccess u, TLinearOperator linear_operator, TRegularizer regularizer, typename TVolumeAccess::elem_t dt)
{
	cuda_run_dual_p_device (u.dim().num_channels, p, u, linear_operator, regularizer, dt);
}
template<int u_num_channels, typename TVolumeAccess, typename TLinearOperator, typename TRegularizer>
__global__ void cuda_run_dual_p_kernel_inline (TVolumeAccess p, TVolumeAccess u, TLinearOperator linear_operator, TRegularizer regularizer, typename TVolumeAccess::elem_t dt)
{
	cuda_run_dual_p_device (u_num_channels, p, u, linear_operator, regularizer, dt);
}
template<typename real>
void DeviceEngine3<real>::run_dual_p(volume_access_t p, volume_access_t u, linear_operator_t linear_operator, regularizer_t regularizer, real dt)
{
	const int u_num_channels = u.dim().num_channels;
    const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
	int sharedmem_p = ShMemArray<real>::size(p_num_channels, block);
	if (u_num_channels == 3)
	{
		cuda_run_dual_p_kernel_inline<3> <<<grid, block, sharedmem_p>>> (p, u, linear_operator, regularizer, dt);  CUDA_CHECK;
	}
	else
	{
		cuda_run_dual_p_kernel <<<grid, block, sharedmem_p>>> (p, u, linear_operator, regularizer, dt);  CUDA_CHECK;
	}
}


template<typename TVolumeAccess, typename TLinearOperator, typename TDataterm>
__device__ void cuda_run_prim_u_device (const int u_num_channels, TVolumeAccess u, TVolumeAccess ubar, TVolumeAccess p, TLinearOperator linear_operator, TDataterm dataterm, typename TVolumeAccess::elem_t theta_bar, typename TVolumeAccess::elem_t dt)
{
	typedef typename TVolumeAccess::elem_t real;

	const Dim3D &dim3d = u.dim().dim3d();
    const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
	int x = cuda_x();
	int y = cuda_y();
	int z = cuda_z();
	if (is_active(x, y, z, dim3d))
	{
		ShMemArray<real> u_sh(u_num_channels);
		ShMemArray<real> valold_sh(u_num_channels, u_sh);

		linear_operator.apply_transpose(u_sh, p, x, y, z, dim3d, u_num_channels);

		for(int i = 0; i < u_num_channels; i++)
		{
			real valold = u.get(x, y, z, i);
			u_sh.get(i) = valold - u_sh.get(i) * dt;
			valold_sh.get(i) = valold;
		}

		dataterm.prox(u_sh, dt, x, y, z, dim3d, u_num_channels);

		for(int i = 0; i < u_num_channels; i++)
		{
			real valnew = u_sh.get(i);
			u.get(x, y, z, i) = valnew;
			real valold = valold_sh.get(i);
			ubar.get(x, y, z, i) = valnew + (valnew - valold) * theta_bar;
		}
	}
}
template<typename TVolumeAccess, typename TLinearOperator, typename TDataterm>
__global__ void cuda_run_prim_u_kernel (TVolumeAccess u, TVolumeAccess ubar, TVolumeAccess p, TLinearOperator linear_operator, TDataterm dataterm, typename TVolumeAccess::elem_t theta_bar, typename TVolumeAccess::elem_t dt)
{
	cuda_run_prim_u_device (u.dim().num_channels, u, ubar, p, linear_operator, dataterm, theta_bar, dt);
}
template<int u_num_channels, typename TVolumeAccess, typename TLinearOperator, typename TDataterm>
__global__ void cuda_run_prim_u_kernel_inline (TVolumeAccess u, TVolumeAccess ubar, TVolumeAccess p, TLinearOperator linear_operator, TDataterm dataterm, typename TVolumeAccess::elem_t theta_bar, typename TVolumeAccess::elem_t dt)
{
	cuda_run_prim_u_device (u_num_channels, u, ubar, p, linear_operator, dataterm, theta_bar, dt);
}
template<typename real>
void DeviceEngine3<real>::run_prim_u(volume_access_t u, volume_access_t ubar, volume_access_t p, linear_operator_t linear_operator, dataterm_t dataterm, real theta_bar, real dt)
{
	int sharedmem_2u = ShMemArray<real>::size(u.dim().num_channels, block) * 2;
	int u_num_channels = u.dim().num_channels;
	if (u_num_channels == 3)
	{
		cuda_run_prim_u_kernel_inline<3> <<<grid, block, sharedmem_2u>>> (u, ubar, p, linear_operator, dataterm, theta_bar, dt);  CUDA_CHECK;
	}
	else
	{
		cuda_run_prim_u_kernel <<<grid, block, sharedmem_2u>>> (u, ubar, p, linear_operator, dataterm, theta_bar, dt);  CUDA_CHECK;
	}
}


template<typename TVolumeAccess, typename TLinearOperator, typename TDataterm, typename TRegularizer>
__global__ void cuda_energy_base_kernel (TVolumeAccess sum, TVolumeAccess u, TLinearOperator linear_operator, TDataterm dataterm, TRegularizer regularizer)
{
	typedef typename TVolumeAccess::elem_t real;

	const Dim3D &dim3d = u.dim().dim3d();
	const int u_num_channels = u.dim().num_channels;
    const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
	int x = cuda_x();
	int y = cuda_y();
	int z = cuda_z();
	if (is_active(x, y, z, dim3d))
	{
		real energy = real(0);

		ShMemArray<real> p_sh(p_num_channels);
		linear_operator.apply(p_sh, u, x, y, z, dim3d, u_num_channels);
		energy += regularizer.value(p_sh, x, y, z, dim3d, p_num_channels);

		__syncthreads();

		ShMemArray<real> u_sh(u_num_channels);
		for(int i = 0; i < u_num_channels; i++)
		{
			u_sh.get(i) = u.get(x, y, z, i);
		}
		energy += dataterm.value(u_sh, x, y, z, dim3d, u_num_channels);

		sum.get(x, y, z, 0) = energy;
	}
}
template<typename real>
void DeviceEngine3<real>::energy_base(volume_access_t u, volume_access_t aux_reduce, linear_operator_t linear_operator, dataterm_t dataterm, regularizer_t regularizer)
{
	const int u_num_channels = u.dim().num_channels;
    const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
	int sharedmem_u = ShMemArray<real>::size(u_num_channels, block);
	int sharedmem_p = ShMemArray<real>::size(p_num_channels, block);
	int sharedmem_max = std::max(sharedmem_u, sharedmem_p);
	cuda_energy_base_kernel <<<grid, block, sharedmem_max>>> (aux_reduce, u, linear_operator, dataterm, regularizer); CUDA_CHECK;
}


template<typename TVolumeAccess, typename TLinearOperator, typename TRegularizer>
__global__ void cuda_add_edges_kernel(TVolumeAccess volume, TLinearOperator linear_operator, TRegularizer regularizer)
{
	typedef typename TVolumeAccess::elem_t real;

	const Dim3D &dim3d = volume.dim().dim3d();
	const int u_num_channels = volume.dim().num_channels;
    const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
	int x = cuda_x();
	int y = cuda_y();
	int z = cuda_z();
	if (is_active(x, y, z, dim3d))
	{
		ShMemArray<real> p_sh(p_num_channels);
		linear_operator.apply(p_sh, volume, x, y, z, dim3d, u_num_channels);
	    const real max_range_norm = linear_operator.maximal_possible_range_norm(u_num_channels);
		real val_edge_indicator = regularizer.edge_indicator(p_sh, max_range_norm, x, y, z, dim3d, p_num_channels);
		real mult = real(1) - val_edge_indicator;
		for (int i = 0; i < u_num_channels; i++)
		{
			volume.get(x, y, z, i) *= mult;
		}
	}
}
template<typename real>
void DeviceEngine3<real>::add_edges(volume_access_t cur_result, linear_operator_t linear_operator, regularizer_t regularizer)
{
	const int p_num_channels = linear_operator.num_channels_range(cur_result.dim().num_channels);
	int sharedmem_p = ShMemArray<real>::size(p_num_channels, block);
	cuda_add_edges_kernel <<<grid, block, sharedmem_p>>> (cur_result, linear_operator, regularizer); CUDA_CHECK;
}


template<typename TVolumeAccess, typename TLinearOperator>
__global__ void cuda_set_regularizer_weight_from__normgrad_kernel (TVolumeAccess regularizer_weight, TVolumeAccess volume, TLinearOperator linear_operator)
{
	typedef typename TVolumeAccess::elem_t real;

	const Dim3D &dim3d = volume.dim().dim3d();
	const int u_num_channels = volume.dim().num_channels;
    const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
	int x = cuda_x();
	int y = cuda_y();
	int z = cuda_z();
	if (is_active(x, y, z, dim3d))
	{
		ShMemArray<real> gradient_sh(p_num_channels);
		linear_operator.apply(gradient_sh, volume, x, y, z, dim3d, u_num_channels);
		regularizer_weight.get(x, y, z, 0) = vec_norm(gradient_sh, p_num_channels);
	}
}
template<typename real>
void DeviceEngine3<real>::set_regularizer_weight_from__normgrad(volume_access_t regularizer_weight, volume_access_t volume, linear_operator_t linear_operator)
{
	const int u_num_channels = volume.dim().num_channels;
    const int p_num_channels = linear_operator.num_channels_range(u_num_channels);
    int sharedmem_gradient = ShMemArray<real>::size(p_num_channels, block);
	cuda_set_regularizer_weight_from__normgrad_kernel <<<grid, block, sharedmem_gradient>>> (regularizer_weight, volume, linear_operator); CUDA_CHECK;
}

template<typename TVolumeAccess>
__global__ void cuda_set_regularizer_weight_from__exp_kernel (TVolumeAccess regularizer_weight, typename TVolumeAccess::elem_t coeff)
{
	typedef typename TVolumeAccess::elem_t real;

	const Dim3D &dim3d = regularizer_weight.dim().dim3d();
	int x = cuda_x();
	int y = cuda_y();
	int z = cuda_z();
	if (is_active(x, y, z, dim3d))
	{
		const real eps = real(1e-6);
		regularizer_weight.get(x, y, z, 0) = realmax(eps, realexp(-coeff * regularizer_weight.get(x, y, z, 0)));
	}
}
template<typename real>
void DeviceEngine3<real>::set_regularizer_weight_from__exp(volume_access_t regularizer_weight, real coeff)
{
	cuda_set_regularizer_weight_from__exp_kernel <<<grid, block>>> (regularizer_weight, coeff); CUDA_CHECK;
}


template<typename TVolumeAccess>
__device__ void cuda_diff_l1_base_device (const int a_num_channels, TVolumeAccess a, TVolumeAccess b, TVolumeAccess aux_reduce)
{
	typedef typename TVolumeAccess::elem_t real;

	const Dim3D &dim3d = a.dim().dim3d();
	int x = cuda_x();
	int y = cuda_y();
	int z = cuda_z();
	if (is_active(x, y, z, dim3d))
	{
		real diff = real(0);
		for (int i = 0; i < a_num_channels; i++)
		{
			real val_a = a.get(x, y, z, i);
			real val_b = b.get(x, y, z, i);
			diff += realabs(val_a - val_b);
		}
		aux_reduce.get(x, y, z, 0) = diff;
	}
}
template<typename TVolumeAccess>
__global__ void cuda_diff_l1_base_kernel (TVolumeAccess a, TVolumeAccess b, TVolumeAccess aux_reduce)
{
	cuda_diff_l1_base_device (a.dim().num_channels, a, b, aux_reduce);
}
template<int a_num_channels, typename TVolumeAccess>
__global__ void cuda_diff_l1_base_kernel_inline (TVolumeAccess a, TVolumeAccess b, TVolumeAccess aux_reduce)
{
	cuda_diff_l1_base_device (a_num_channels, a, b, aux_reduce);
}
template<typename real>
void DeviceEngine3<real>::diff_l1_base(volume_access_t a, volume_access_t b, volume_access_t aux_reduce)
{
	const int a_num_channels = a.dim().num_channels;
	if (a_num_channels == 3)
	{
		cuda_diff_l1_base_kernel_inline<3> <<<grid, block>>> (a, b, aux_reduce);  CUDA_CHECK;
	}
	else
	{
		cuda_diff_l1_base_kernel <<<grid, block>>> (a, b, aux_reduce);  CUDA_CHECK;
	}
}




template<typename real>
class VolumeSolverDeviceImplementation: public VolumeSolverBase<real>
{
public:
	VolumeSolverDeviceImplementation() { VolumeSolverBase<real>::set_engine(&engine);	}
private:
	DeviceEngine3<real> engine;
};


template<typename real> VolumeSolverDevice<real>::VolumeSolverDevice() : implementation(NULL) { implementation = new VolumeSolverDeviceImplementation<real>(); }
template<typename real> VolumeSolverDevice<real>::~VolumeSolverDevice() { delete implementation; }
template<typename real> BaseVolume* VolumeSolverDevice<real>::run(const BaseVolume *volume, const Par3 &par) { return implementation->run(volume, par); }
template class VolumeSolverDevice<float>;
template class VolumeSolverDevice<double>;



#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)

