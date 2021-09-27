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

#ifndef VOLUME_SOLVER_BASE_H
#define VOLUME_SOLVER_BASE_H

#include "volume_solver_common_operators.h"
#include "util/volume.h"



template<typename real>
class Engine3
{
public:
	typedef VolumeAccess<real, DataInterpretationLayered> volume_access_t;
	typedef typename volume_access_t::data_interpretation_t data_interpretation_t;
	typedef LinearOperator3<real> linear_operator_t;
	typedef Regularizer3<volume_access_t> regularizer_t;
	typedef Dataterm3<volume_access_t> dataterm_t;
	typedef VolumeManagerBase<real, data_interpretation_t> volume_manager_base_t;

	virtual ~Engine3() {}
	virtual std::string str() { return ""; };
	virtual void alloc(const ArrayDim3 &dim_u) = 0;
	virtual void free() = 0;
	virtual bool is_valid() = 0;
	virtual volume_manager_base_t* volume_manager() = 0;
	virtual real get_sum(volume_access_t a) = 0;
	virtual void timer_start() = 0;
	virtual void timer_end() = 0;
	virtual double timer_get() = 0;
	virtual void synchronize() = 0;

	virtual void run_dual_p(volume_access_t p, volume_access_t u, linear_operator_t linear_operator, regularizer_t regularizer, real dt) = 0;
	virtual void run_prim_u(volume_access_t u, volume_access_t ubar, volume_access_t p, linear_operator_t linear_operator, dataterm_t dataterm, real theta_bar, real dt) = 0;
	virtual void energy_base(volume_access_t u, volume_access_t aux_reduce, linear_operator_t linear_operator, dataterm_t dataterm, regularizer_t regularizer) = 0;
	virtual void add_edges(volume_access_t cur_result, linear_operator_t linear_operator, regularizer_t regularizer) = 0;
	virtual void set_regularizer_weight_from__normgrad(volume_access_t regularizer_weight, volume_access_t volume, linear_operator_t linear_operator) = 0;
	virtual void set_regularizer_weight_from__exp(volume_access_t regularizer_weight, real coeff) = 0;
	virtual void diff_l1_base(volume_access_t a, volume_access_t b, volume_access_t aux_reduce) = 0;
};


template<typename real>
class VolumeSolverBase
{
public:
	VolumeSolverBase();
	virtual ~VolumeSolverBase();


	BaseVolume* run(const BaseVolume *volume, const Par3 &par_const);

protected:
	void set_engine(Engine3<real> *engine);

private:
	typedef typename Engine3<real>::volume_access_t volume_access_t;
	typedef typename Engine3<real>::linear_operator_t linear_operator_t;

	size_t alloc(const ArrayDim3 &dim_u);
	void free();
	void init(const BaseVolume *volume);
	void set_regularizer_weight_from(volume_access_t volume);
	real energy();
	real diff_l1(volume_access_t a, volume_access_t b);
	bool is_converged(int iteration);
	void print_stats();
	BaseVolume* get_solution(const BaseVolume *volume);

	Engine3<real> *engine;
	Par3 par;
	PrimalDualVars3<volume_access_t> pd_vars;
	bool u_is_computed;

	struct Arrays
	{
		size_t alloc(Engine3<real> *engine, const ArrayDim3 &dim_u, const ArrayDim3 &dim_p)
		{
			// TODO: MULTICHANNEL
			ArrayDim3 dim_scalar(dim_u.w, dim_u.h, dim_u.d, 1);
			size_t mem = 0;
			mem += engine->volume_manager()->alloc(u, dim_u);
			mem += engine->volume_manager()->alloc(ubar, dim_u);
			mem += engine->volume_manager()->alloc(f, dim_u);
			mem += engine->volume_manager()->alloc(p, dim_p);
			mem += engine->volume_manager()->alloc(regularizer_weight, dim_scalar);
			mem += engine->volume_manager()->alloc(prev_u, dim_u);
			mem += engine->volume_manager()->alloc(aux_result, dim_u);
			mem += engine->volume_manager()->alloc(aux_reduce, dim_scalar);
			return mem;
		}
		void free(Engine3<real> *engine)
		{
			engine->volume_manager()->free(u);
			engine->volume_manager()->free(ubar);
			engine->volume_manager()->free(f);
			engine->volume_manager()->free(p);
			engine->volume_manager()->free(regularizer_weight);
			engine->volume_manager()->free(prev_u);
			engine->volume_manager()->free(aux_result);
			engine->volume_manager()->free(aux_reduce);
		}
		volume_access_t u;
		volume_access_t ubar;
		volume_access_t f;
		volume_access_t p;
		volume_access_t regularizer_weight;
		volume_access_t prev_u;
		volume_access_t aux_result;
		volume_access_t aux_reduce;
	} arr;

	struct ResultStats
	{
		ResultStats()
		{
			mem = 0;
			stop_iteration = -1;
			time_compute = 0.0;
			time_compute_sum = 0.0;
			time = 0.0;
			time_sum = 0.0;
			num_runs = 0;
			energy = real(0);
		}
		ArrayDim3 dim_u;
		ArrayDim3 dim_p;
		size_t mem;
		int stop_iteration;
		double time_compute;      // actual computation without alloc and volume conversions and copying, in seconds
		double time_compute_sum;  // accumulation for averaging
		double time;              // overall time, including alloc, volume conversions and copying, in second
		double time_sum;          // accumulation for averaging
		int num_runs;
		real energy;
	} stats;
};


#endif // SOLVER_BASE_H
