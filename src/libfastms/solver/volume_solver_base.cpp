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

#include "volume_solver_base.h"
#include <cstdio>  // for snprintf
#include "util/timer.h"



template<typename real>
VolumeSolverBase<real>::VolumeSolverBase()
{
	engine = NULL;
	u_is_computed = false;
}


template<typename real>
VolumeSolverBase<real>::~VolumeSolverBase()
{
}


template<typename real>
void VolumeSolverBase<real>::set_engine(Engine3<real> *engine)
{
	this->engine = engine;
}


template<typename real>
size_t VolumeSolverBase<real>::alloc(const ArrayDim3 &dim_u)
{
	engine->alloc(dim_u);
	const ArrayDim3 &dim_p = pd_vars.linear_operator.dim_range(dim_u);
	size_t mem = arr.alloc(engine, dim_u, dim_p);
	if (mem > 0) { u_is_computed = false; }
	return mem;
}


template<typename real>
void VolumeSolverBase<real>::free()
{
	arr.free(engine);
	engine->free();
}


template<typename real>
void VolumeSolverBase<real>::init(const BaseVolume *volume)
{
	volume->copy_to_layered(arr.f.get_untyped_access());
	if (par.temporal == real(0)) { u_is_computed = false; }
	if (u_is_computed)
	{
		engine->volume_manager()->copy_from_samekind(arr.prev_u, arr.u);
	}
	engine->volume_manager()->copy_from_samekind(arr.u, arr.f);
	engine->volume_manager()->copy_from_samekind(arr.ubar, arr.u);
	engine->volume_manager()->setzero(arr.p);
    if (par.weight)
    {
	    set_regularizer_weight_from(arr.f);
    }
    pd_vars.init(par, arr.f, arr.regularizer_weight, (u_is_computed? arr.prev_u : volume_access_t()));
}


template<typename real>
void VolumeSolverBase<real>::set_regularizer_weight_from(volume_access_t volume)
{
	linear_operator_t linear_operator;
	const Dim3D &dim3d = volume.dim().dim3d();

	// real gamma = real(1);
    engine->set_regularizer_weight_from__normgrad(arr.regularizer_weight, volume, linear_operator);
	real sigma = engine->get_sum(arr.regularizer_weight) / (real(dim3d.w) * real(dim3d.h));

    real coeff = (sigma > real(0)? real(2) / sigma : real(0));  // 2 = dim_volume_domain
    engine->set_regularizer_weight_from__exp (arr.regularizer_weight, coeff);
}


template<typename real>
real VolumeSolverBase<real>::energy()
{
	engine->energy_base(arr.u, arr.aux_reduce, pd_vars.linear_operator, pd_vars.dataterm, pd_vars.regularizer);
	real energy = engine->get_sum(arr.aux_reduce);
    real mult = real(1) / (pd_vars.scale_omega * pd_vars.scale_omega);
	energy *= mult;
    return energy;
}


template<typename real>
real VolumeSolverBase<real>::diff_l1(volume_access_t a, volume_access_t b)
{
	engine->diff_l1_base(a, b, arr.aux_reduce);
	real diff = engine->get_sum(arr.aux_reduce);
	const Dim3D &dim3d = a.dim().dim3d();
	diff /= (size_t)dim3d.w * dim3d.h;
	return diff;
}
template<typename real>
bool VolumeSolverBase<real>::is_converged(int iteration)
{
	if (par.stop_k <= 0 || (iteration + 1) % par.stop_k != 0)
	{
		return false;
	}
	real diff_to_prev = diff_l1(arr.u, arr.ubar) / pd_vars.theta_bar;
	return (diff_to_prev <= par.stop_eps);
}


template<typename real>
BaseVolume* VolumeSolverBase<real>::get_solution(const BaseVolume *volume)
{
	engine->volume_manager()->copy_from_samekind(arr.aux_result, arr.u);
	if (par.edges)
	{
		engine->add_edges(arr.aux_result, pd_vars.linear_operator, pd_vars.regularizer);
	}
    BaseVolume* out_volume = volume->new_of_same_type_and_size();
	out_volume->copy_from_layered(arr.aux_result.get_untyped_access());
    return out_volume;
}


template<typename real>
void VolumeSolverBase<real>::print_stats()
{
	if (stats.mem > 0)
	{
		std::cout << "alloc " << (stats.mem + (1<<20) - 1) / (1<<20) << " MB for ";
		std::cout << stats.dim_u << ",  ";
	}
	std::string str_from_engine = engine->str();
	if (str_from_engine != "") { std::cout << str_from_engine.c_str() << ", "; }
	char buffer[100];
	snprintf(buffer, sizeof(buffer), "%2.4f s compute / %2.4f s all (+ %2.4f)", stats.time_compute, stats.time, stats.time - stats.time_compute); std::cout << buffer;
	if (stats.num_runs > 1)
	{
		snprintf(buffer, sizeof(buffer), ", average %2.4f s / %2.4f s (+ %2.4f)", stats.time_compute_sum / stats.num_runs, stats.time_sum / stats.num_runs, (stats.time_sum - stats.time_compute_sum) / stats.num_runs); std::cout << buffer;
	}
	if (stats.stop_iteration != -1)
	{
		std::cout << ", " << (stats.stop_iteration + 1) << " iterations";
	}
	else
	{
		std::cout << ", did not stop after " << par.iterations << " iterations";
	}
	std::cout << ", lambda " << par.lambda;
	if (par.adapt_params) { std::cout << " (adapted " << pd_vars.regularizer.lambda << ")"; }
	std::cout << ", alpha " << par.alpha;
	if (par.adapt_params) { std::cout << " (adapted " << pd_vars.regularizer.alpha << ")"; }
	if (par.temporal > 0)
	{
		std::cout << ", temporal " << par.temporal;
	}
	if (par.weight)
	{
		std::cout << ", weighting";
	}
	std::cout << ", energy ";
	snprintf(buffer, sizeof(buffer), "%4.4f", stats.energy); std::cout << buffer;
	std::cout << std::endl;
}


template<typename real>
BaseVolume* VolumeSolverBase<real>::run(const BaseVolume *volume, const Par3 &par_const)
{
	if (!engine->is_valid()) { BaseVolume *out_volume = volume->new_of_same_type_and_size(); return out_volume; }
	Timer timer_all;
	timer_all.start();

	// allocate (only if not already allocated)
	this->par = par_const;
	stats.dim_u = volume->dim();
	stats.dim_p = linear_operator_t::dim_range(stats.dim_u);
    stats.mem = alloc(stats.dim_u);


    // initialize
	init(volume);


	// compute
	engine->timer_start();
    stats.stop_iteration = -1;
    for (int iteration = 0; iteration < par.iterations; iteration++)
    {
    	pd_vars.update_vars();
		if (iteration == 1200) std::cout << "here" << std::endl;
    	engine->run_dual_p(arr.p, arr.ubar, pd_vars.linear_operator, pd_vars.regularizer, pd_vars.dt_d);
    	engine->run_prim_u(arr.u, arr.ubar, arr.p, pd_vars.linear_operator, pd_vars.dataterm, pd_vars.theta_bar, pd_vars.dt_p);
    	if (is_converged(iteration)) { stats.stop_iteration = iteration; break; }
    }
    engine->timer_end();
    u_is_computed = true;
    stats.time_compute = engine->timer_get();
    stats.time_compute_sum += stats.time_compute;
    stats.num_runs++;
    stats.energy = energy();


    // get solution
    BaseVolume *result = get_solution(volume);
    engine->synchronize();
    timer_all.end();
    stats.time = timer_all.get();
    stats.time_sum += stats.time;
    if (par.verbose) { print_stats(); }
    return result;
}


template class VolumeSolverBase<float>;
template class VolumeSolverBase<double>;
