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

#ifndef VOLUME_SOLVER_COMMON_OPERATORS_HCU
#define VOLUME_SOLVER_COMMON_OPERATORS_HCU

#include "volume_solver.h"
#include "util/volume_access.h"

#if !defined(DISABLE_CUDA) && defined(__CUDACC__)
#define HOST_DEVICE __host__ __device__
#define FORCEINLINE __forceinline__
#else
#define HOST_DEVICE
#define FORCEINLINE inline
#endif // !defined(DISABLE_CUDA) && defined(__CUDACC__)



// Gradient operator with forward differences in indep dims and channels 
template<typename real>
class LinearOperator3
{
public:
	HOST_DEVICE static int num_channels_range(int num_channels_domain)
	{
		return 3 * num_channels_domain;
	}

	HOST_DEVICE static ArrayDim3 dim_range(const ArrayDim3 &dim_domain)
	{
		ArrayDim3 dim_range_result = dim_domain;
		dim_range_result.num_channels = num_channels_range(dim_domain.num_channels);
		return dim_range_result;
	}

	HOST_DEVICE static real maximal_possible_range_norm(int num_channels_domain)
	{
		return realsqrt(real(3)) * num_channels_range(num_channels_domain);
	}

	template<typename TArray, typename TAccess>
	HOST_DEVICE void apply(TArray &p, TAccess &u, int x, int y, int z, const Dim3D &dim3d, const int u_num_channels)
	{
		for(int i = 0; i < u_num_channels; i++)
		{
			real u0 = u.get(x, y, z, i);
			// Get forward difference - zero padded! 
			real uplusx = (x + 1 < dim3d.w? u.get(x + 1, y, z, i) : real(0));
			real uplusy = (y + 1 < dim3d.h? u.get(x, y + 1, z, i) : real(0));
			real uplusz = (z + 1 < dim3d.d? u.get(x, y, z + 1, i): real(0));
			p.get(0 + 3 * i) = (x + 1 < dim3d.w? uplusx - u0 : real(0));
			p.get(1 + 3 * i) = (y + 1 < dim3d.h? uplusy - u0 : real(0));
			p.get(2 + 3 * i) = (z + 1 < dim3d.d? uplusz - u0 : real(0));
		}
	}

	// initial value of sigma in PD descent
	HOST_DEVICE real apply_sumcoeffs()
	{
		return real(2);
	}

	// Transpose version of gradient op, dot prod?
	template<typename Array1D, typename TAccess>
	HOST_DEVICE void apply_transpose(Array1D &u, TAccess &p, int x, int y, int z, const Dim3D &dim3d, const int u_num_channels)
	{
		for(int i = 0; i < u_num_channels; i++)
		{
			real p1_0 = (x + 1 < dim3d.w? p.get(x, y, z, 0 + 3 * i) : real(0));
			real p1_x = (x > 0? p.get(x - 1, y, z, 0 + 3 * i) : real(0));
			
			real p2_0 = (y + 1 < dim3d.h? p.get(x, y, z, 1 + 3 * i) : real(0));
			real p2_y = (y > 0? p.get(x, y - 1, z, 1 + 3 * i) : real(0));
			
			real p3_0 = (z > 0? p.get(x, y, z, 2 + 3 * i) : real(0));
			real p3_z = (z > 0? p.get(x, y - 1, z - 1, 2 + 3 * i) : real(0));
			real val = p1_x - p1_0 + p2_y - p2_0  + p3_z - p3_0;
			u.get(i) = val;
		}
	}

	// Initial value of tau in PD descent for 3 dims
	HOST_DEVICE real apply_transpose_sumcoeffs()
	{
		return real(6);
	}
};


// Quadratic data term (u - f) ^ 2
template<typename TVolumeAccess>
class Dataterm3
{
public:
	typedef typename TVolumeAccess::elem_t real;

	HOST_DEVICE Dataterm3() : temporal(real(0))
	{
	}

	// the coefficient c in the data term: c * (u-f) ^ 2, usually c = 1.
	HOST_DEVICE real get_coeff()
	{
		return real(1);
	}

    HOST_DEVICE bool has_temporal()
    {
    	return prev_u.is_valid() && (temporal > real(0) || temporal < real(0));
    }

	template<typename Array1D>
	HOST_DEVICE void prox (Array1D &u, real dt, int x, int y, int z, const Dim3D &dim3d, const int u_num_channels)
	{
		// arg min_u  (u - u0)^2 / (2 * dt)  +  coeff * (u - f)^2
		real c0 = get_coeff();

		for(int i = 0; i < u_num_channels; i++)
		{
			real f0 = f.get(x, y, z, i);
			real u0 = u.get(i);
			u0 = f0 + (u0 - f0) / (real(1) + real(2) * dt * c0);
			u.get(i) = u0;
		}
		//TODO: Temporal support
		if (has_temporal())
		{
			for (int i = 0; i < u_num_channels; i++) { u.get(i) -= prev_u.get(x, y, z, i); }

			real nrm = vec_norm(u, u_num_channels);
			if (nrm > real(0))
			{
				// default value for temporal = infinity
				real mult = real(0);
				if (temporal > real(0) && temporal < realmax<real>())
				{
					real gamma = temporal * dt / (real(1) + real(2) * dt * c0);
					real a = gamma * real(1.5) / realsqrt(nrm);
					mult = real(2) / (a + realsqrt(a * a + real(4)));
					mult = mult * mult;
				}
				vec_scale_eq(u, u_num_channels, mult);
			}

			for(int i = 0; i < u_num_channels; i++) { u.get(i) += prev_u.get(x, y, z, i); }
		}
	}

	template<typename Array1D>
	HOST_DEVICE real value (Array1D &u, int x, int y, int z, const Dim3D &dim3d, const int u_num_channels)
	{
		// coeff * (u - f)^2
		real c0 = get_coeff();

		real val = real(0);

		real diff_f = real(0);
		for(int i = 0; i < u_num_channels; i++)
		{
			real diff = u.get(i) - f.get(x, y, z, i);
			diff_f += diff * diff;
		}
		diff_f *= c0;
		val += diff_f;

		//TODO: TEMPORAL Support
		if (has_temporal())
		{
			real diff_prev = 0;
			for(int i = 0; i < u_num_channels; i++)
			{
				real diff = u.get(i) - prev_u.get(x, y, z, i);
				diff_prev += diff * diff;
			}
			diff_prev = realsqrt(diff_prev);
			diff_prev = temporal * diff_prev * realsqrt(diff_prev);
			val += diff_prev;
		}

		return val;
	}

	TVolumeAccess f;
	TVolumeAccess prev_u;
	real temporal;
};


// Mumford-Shah regularizer min(alpha * |nabla u| ^ 2, lambda)
template<typename TVolumeAccess>
class Regularizer3
{
public:
	typedef typename TVolumeAccess::elem_t real;

	HOST_DEVICE Regularizer3() : lambda(0), alpha(0)
	{
	}

	template<typename Array1D>
	HOST_DEVICE void prox_star(Array1D &p, real dt, int x, int y, int z, const Dim3D &dim3d, const int p_num_channels)
	{
		real weight0 = (weight.is_valid()? weight.get(x, y, z, 0) : real(1));
		// min(alpha * |g|^2, lambda * weight)
		real nrm2 = vec_norm_squared(p, p_num_channels);
		real A = (alpha >= 0 && alpha < realmax<real>()? real(2) * alpha / (dt + real(2) * alpha) : real(1));
		real L = (lambda >= 0 && lambda < realmax<real>()? real(2) * dt * lambda * weight0 : realmax<real>());
		real mult = (nrm2 * A <= L? A : real(0));
    	vec_scale_eq (p, p_num_channels, mult);
	}

	template<typename Array1D>
	HOST_DEVICE real value(Array1D &p, int x, int y, int z, const Dim3D &dim3d, const int p_num_channels)
	{
		real weight0 = (weight.is_valid()? weight.get(x, y, z, 0) : real(1));
		real A = (alpha >= 0 && alpha < realmax<real>()? alpha : realmax<real>());
		real L = (lambda >= 0 && lambda < realmax<real>()? lambda * weight0 : realmax<real>());
		real nrm = vec_norm(p, p_num_channels);
		real val = real(0);
		if (A < realmax<real>())
		{
			val = realmin(A * nrm * nrm, L);
		}
		else
		{
			real eps = (alpha < real(0)? real(1e-6) : -alpha);
			val = (nrm > eps? L : real(0));
		}
		return val;
	}

	template<typename Array1D>
	HOST_DEVICE real edge_indicator(Array1D &p, real max_range_norm, int x, int y, int z, const Dim3D &dim3d, const int p_num_channels)
	{
		real weight0 = (weight.is_valid()? weight.get(x, y, z, 0) : real(1));
		real A = (alpha >= 0 && alpha < realmax<real>()? alpha : realmax<real>());
		real L = (lambda >= 0 && lambda < realmax<real>()? lambda * weight0 : realmax<real>());
		// Pixel (x,y) is an edge pixel if the gradient is so large that in
		// min(alpha * |gradient| ^ 2, lambda * weight) = min(A * |gradient| ^ 2, L)
		// the minimum is achieved by the second argument.
		real norm_threshold = realsqrt(L / A);
		const real eps = real(5e-3);
		//if (A >= realmax<real>()) norm_threshold = eps;
		norm_threshold = realmax(norm_threshold, eps);

		real cur_norm = vec_norm(p, p_num_channels);
		max_range_norm = realmax(max_range_norm, cur_norm);  // to be sure
		real val_edge_indicator = real(0);
		if (cur_norm > norm_threshold)
		{
			// It holds 1 < cur_norm / norm_threshold.
			// Because of max_range_norm >= cur_norm: Also 1 < cur_norm / norm_threshold <= max_range_norm / norm_threshold.
			// Therefore 0 < log(cur_norm / norm_threshold) <= log(max_range_norm / norm_threshold),
			// i.e. 0 < log(cur_norm / norm_threshold) / log(max_range_norm / norm_threshold) <= 1.
			// Use this as edge indicator.
			val_edge_indicator = reallog(cur_norm / norm_threshold) / reallog(max_range_norm / norm_threshold);
		}
		return val_edge_indicator;
	}

	real lambda;
	real alpha;
	TVolumeAccess weight;
};


template<typename TVolumeAccess>
class PrimalDualVars3
{
public:
	typedef typename TVolumeAccess::elem_t real;

	PrimalDualVars3()
	{
		dt_p = real(0);
		dt_d = real(0);
		theta_bar = real(0);
		gamma_dataterm = real(0);
		scale_omega = real(0);
	}

	void init(const Par3 &par, TVolumeAccess f, TVolumeAccess regularizer_weight, TVolumeAccess prev_u)
	{
		real dt_factor = real(1);
		dt_p = real(1) * dt_factor / linear_operator.apply_transpose_sumcoeffs();
		dt_d = real(1) / dt_factor / linear_operator.apply_sumcoeffs();
	    theta_bar = real(1);
	    gamma_dataterm = real(2);
	    if (par.adapt_params)
	    {
	    	if (f.dim().h > 1)
	    	{
			    scale_omega = realsqrt(real(f.dim().w) * real(f.dim().h) * real(f.dim().d)) / realsqrt(real(500) * real(500) * real(500));
	    	}
	    	else
	    	{
	    		scale_omega = real(f.dim().w) / real(500);
	    	}
	    }
	    else
	    {
	    	scale_omega = real(1);
	    }

	    dataterm.f = f;
	    bool has_temporal = (prev_u.is_valid() && (par.temporal > real(0) || par.temporal < real(0)));
	    dataterm.prev_u = (has_temporal? prev_u : TVolumeAccess());
	    dataterm.temporal = (has_temporal? par.temporal : real(0));
	    regularizer.alpha = (par.adapt_params && par.alpha >= 0 && par.alpha < realmax<real>()? par.alpha * scale_omega * scale_omega : par.alpha);
	    regularizer.lambda = (par.adapt_params && par.lambda >= 0 && par.lambda < realmax<real>()? par.lambda * scale_omega : par.lambda);
	    regularizer.weight = (par.weight? regularizer_weight : TVolumeAccess());
	}

	void update_vars()
	{
    	dt_p *= theta_bar;
    	dt_d /= theta_bar;
    	theta_bar = real(1) / realsqrt(real(1) + real(2) * gamma_dataterm * dt_p);
	}

	real dt_p;
	real dt_d;
	real theta_bar;
	real gamma_dataterm;
    real scale_omega;

    LinearOperator3<real> linear_operator;
    Dataterm3<TVolumeAccess> dataterm;
    Regularizer3<TVolumeAccess> regularizer;
};



#undef HOST_DEVICE
#undef FORCEINLINE

#endif // VOLUME_SOLVER_COMMON_OPERATORS_HCU
