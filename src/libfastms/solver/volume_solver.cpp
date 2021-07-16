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

#include "volume_solver.h"

#include "volume_solver_host.h"
#ifndef DISABLE_CUDA
#include "volume_solver_device.h"
#endif // not DISABLE_CUDA

#include "util/volume.h"
#include "util/types_equal.h"
#include "util/has_cuda.h"
#include <iostream>



class SolverImplementation3
{
public:
	virtual ~SolverImplementation3() {}

	// general
	virtual BaseVolume* run(const BaseVolume *in, const Par3 &par) = 0;

	// layered real
	virtual void run(float *&out_volume, const float *in_volume, const ArrayDim3 &dim, const Par3 &par) = 0;
	virtual void run(double *&out_volume, const double *in_volume, const ArrayDim3 &dim, const Par3 &par) = 0;

	// interlaced char
	virtual void run(unsigned char *&out_volume, const unsigned char *in_volume, const ArrayDim3 &dim, const Par3 &par) = 0;

	virtual int get_class_type() = 0; // for is_instance_of()
};



namespace
{

template<typename Solver3>
class SolverImplementationConcrete: public SolverImplementation3
{
public:
	virtual ~SolverImplementationConcrete() {};

	// general
	virtual BaseVolume* run(const BaseVolume *in, const Par3 &par)
	{
		return volume_solver.run(in, par);
	}

	// layered real
	virtual void run(float *&out_volume, const float *in_volume, const ArrayDim3 &dim, const Par3 &par)
	{
		run_real(out_volume, in_volume, dim, par);
	}
	virtual void run(double *&out_volume, const double *in_volume, const ArrayDim3 &dim, const Par3 &par)
	{
		run_real(out_volume, in_volume, dim, par);
	}
	template<typename real> void run_real(real *&out_volume, const real *in_volume, const ArrayDim3 &dim, const Par3 &par)
	{
		typedef ManagedVolume<real, DataInterpretationLayered> managed_volume_t;

		managed_volume_t in_managed(const_cast<real*>(in_volume), dim);
		managed_volume_t *out_managed = static_cast<managed_volume_t*>(volume_solver.run(&in_managed, par));
		if (out_volume)
		{
			// copy
			managed_volume_t outimage_managed(out_volume, dim);
			outimage_managed.copy_from_samekind(out_managed);
		}
		else
		{
			// move
			out_volume = out_managed->release_data();
		}
		delete out_managed;
	}

	// interlaced char
	virtual void run(unsigned char *&out_volume, const unsigned char *in_volume, const ArrayDim3 &dim, const Par3 &par)
	{
		typedef ManagedVolume<unsigned char, DataInterpretationInterlaced> managed_volume_t;

		managed_volume_t in_managed(const_cast<unsigned char*>(in_volume), dim);
		managed_volume_t *out_managed = static_cast<managed_volume_t*>(volume_solver.run(&in_managed, par));
		if (out_volume)
		{
			// copy
			managed_volume_t outimage_managed(out_volume, dim);
			outimage_managed.copy_from_samekind(out_managed);
		}
		else
		{
			// move
			out_volume = out_managed->release_data();
		}
		delete out_managed;
	}


	int get_class_type() { return class_type; }
	static int static_get_class_type() { return class_type; }

private:
	Solver3 volume_solver;

	static const int class_type =
#ifndef DISABLE_CUDA
	(types_equal<Solver3, VolumeSolverHost<float> >::value? 0 : \
     types_equal<Solver3, VolumeSolverHost<double> >::value? 1 : \
     types_equal<Solver3, VolumeSolverDevice<float> >::value? 2 : \
     types_equal<Solver3, VolumeSolverDevice<double> >::value? 3 : -1);
#else
	(types_equal<Solver3, VolumeSolverHost<float> >::value? 0 : \
     types_equal<Solver3, VolumeSolverHost<double> >::value? 1 : -1);
#endif // not DISABLE_CUDA
};


template<typename T_class, typename T_object> bool is_instance_of(T_object *object)
{
	return (T_class::static_get_class_type() == object->get_class_type());
}
template<typename Implementation> void set_implementation_concrete(SolverImplementation3 *&implementation, const Par3 &par)
{
	if (implementation && !is_instance_of<Implementation>(implementation))
	{
		// allocated, but wrong class
		delete implementation;
		implementation = NULL;
	}
	if (!implementation)
	{
		implementation = new Implementation();
	}
}
template<typename real> void set_implementation_real(SolverImplementation3 *&implementation, const Par3 &par)
{
	switch (par.engine)
	{
		case Par3::engine_cpu:
		{
			set_implementation_concrete<SolverImplementationConcrete<VolumeSolverHost<real> > >(implementation, par);
			return;
		}
		case Par3::engine_cuda:
		{
			std::string error_str;
			bool cuda_ok = has_cuda(&error_str);
#ifndef DISABLE_CUDA
			if (cuda_ok) { set_implementation_concrete<SolverImplementationConcrete<VolumeSolverDevice<real> > >(implementation, par); }
#endif // not DISABLE_CUDA
			if (!cuda_ok)
			{
				std::cerr << "ERROR: Solver3::run(): Could not select CUDA engine, USING CPU VERSION INSTEAD (" << error_str.c_str() << ")." << std::endl;
				Par3 par_cpu = par;
				par_cpu.engine = Par3::engine_cpu;
				set_implementation_real<real>(implementation, par_cpu);
			}
			break;
		}
		default:
		{
			std::cerr << "ERROR: Solver3::run(): Unexpected engine " << par.engine << ", USING CPU VERSION INSTEAD" << std::endl;
			Par3 par_cpu = par;
			par_cpu.engine = Par3::engine_cpu;
			set_implementation_real<real>(implementation, par_cpu);
		}
	}
}
void set_implementation(SolverImplementation3 *&implementation, const Par3 &par)
{
	if (par.use_double)
	{
		set_implementation_real<double>(implementation, par);
	}
	else
	{
		set_implementation_real<float>(implementation, par);
	}
}

} // namespace



Solver3::Solver3() : implementation(NULL) {}
Solver3::~Solver3() { if (implementation) { delete implementation; } }
BaseVolume* Solver3::run(const BaseVolume *in, const Par3 &par)
{
	set_implementation(implementation, par); if (!implementation) { return NULL; }
	return implementation->run(in, par);
}
void Solver3::run(float *&out_volume, const float *in_volume, const ArrayDim3 &dim, const Par3 &par)
{
	set_implementation_real<float>(implementation, par); if (!implementation) { return; }
	return implementation->run(out_volume, in_volume, dim, par);
}
void Solver3::run(double *&out_volume, const double *in_volume, const ArrayDim3 &dim, const Par3 &par)
{
	set_implementation_real<double>(implementation, par); if (!implementation) { return; }
	return implementation->run(out_volume, in_volume, dim, par);
}
void Solver3::run(unsigned char *&out_volume, const unsigned char *in_volume, const ArrayDim3 &dim, const Par3 &par)
{
	set_implementation(implementation, par); if (!implementation) { return; }
	return implementation->run(out_volume, in_volume, dim, par);
}

