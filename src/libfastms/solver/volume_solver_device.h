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

#ifndef VOLUME_SOLVER_DEVICE_H
#define VOLUME_SOLVER_DEVICE_H

#ifndef DISABLE_CUDA

#include "volume_solver.h"



template<typename real> class VolumeSolverDeviceImplementation;

template<typename real>
class VolumeSolverDevice
{
public:
	typedef real real_t;

	VolumeSolverDevice();
	~VolumeSolverDevice();

	BaseVolume* run(const BaseVolume *volume, const Par &par);

private:
	VolumeSolverDevice(const VolumeSolverDevice<real> &other_solver);  // disable
	VolumeSolverDevice<real>& operator= (const VolumeSolverDevice<real> &other_solver);  // disable

	VolumeSolverDeviceImplementation<real> *implementation;
};


#endif // not DISABLE_CUDA

#endif // SOLVER_DEVICE_H
