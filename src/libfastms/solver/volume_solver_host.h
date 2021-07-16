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

#ifndef VOLUME_SOLVER_HOST_H
#define VOLUME_SOLVER_HOST_H

#include "volume_solver.h"


template<typename real> class VolumeSolverHostImplementation;

template<typename real>
class VolumeSolverHost
{
public:
	typedef real real_t;

	VolumeSolverHost();
	~VolumeSolverHost();

	BaseVolume* run(const BaseVolume *in_volume, const Par3 &par);

private:
	VolumeSolverHost(const VolumeSolverHost<real> &other_solver);  // disable
	VolumeSolverHost<real>& operator= (const VolumeSolverHost<real> &other_solver);  // disable

	VolumeSolverHostImplementation<real> *implementation;
};



#endif // SOLVER_HOST_H
