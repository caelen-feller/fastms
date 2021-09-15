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

#ifndef VOLUME_SOLVER_H
#define VOLUME_SOLVER_H

#include <iostream>



struct Par3
{
	Par3()
	{
		lambda = 0.1;
		alpha = 20.0;
		temporal = 0.0;
		iterations = 10000;
		stop_eps = 5e-5;
		stop_k = 10;
		adapt_params = false;
		weight = false;
		edges = false;
		use_double = false;
		engine = engine_cuda;
		verbose = true;
	}
	// VolMat
	VolMat run(const VolMat in_image, const Par3 &par);

	void print() const
	{
	    std::cout << "Params:\n";
	    std::cout << "  lambda: " << lambda << "\n";
	    std::cout << "  alpha: " << alpha << "\n";
	    std::cout << "  temporal: " << temporal << "\n";
	    std::cout << "  iterations: " << iterations << "\n";
	    std::cout << "  stop_eps: " << stop_eps << "\n";
	    std::cout << "  stop_k: " << stop_k << "\n";
	    std::cout << "  adapt_params: " << adapt_params << "\n";
	    std::cout << "  weight: " << weight << "\n";
	    std::cout << "  edges: " << edges << "\n";
	    std::cout << "  use_double: " << use_double << "\n";
	    std::cout << "  engine: " << (engine == Par3::engine_cpu? "cpu" : "cuda") << "\n";
	}

	// Length penalization parameter.
	// For bigger values:
	//   Number of discontinuities will be smaller, i.e. the solution will be smooth over larger regions.
	//   To set lambda = infinity: Set lambda to any value < 0. Solution will be a constant one, with one and the same color in every pixel ( = mean color).
	// For smaller values:
	//   More discontinuities will arise, i.e. the regions of smoothness will be smaller and the solution will resemble the input image more and more.
	//   Value lambda = 0: Solution will be equal to the original input image.
    double lambda;

    // Smoothness penalization parameter.
    // For bigger values:
    //   Solution will be more "flat" in the regions of smoothness (between the color edges), i.e. there will be only a small change in color values from pixel to pixel.
    //   To set alpha = infinity: Set alpha to any value < 0. This is the cartoon limit case, i.e. the solution will be piecewise constant. This is the segmentation case.
    // For smaller values:
    //   Solution will be more "rough" in the regions of smoothness (between the color edges), i.e. the color values are allowed to change more rapidly from pixel to pixel.
    //   Value alpha = 0: Solution will be equal to the original input image.
    double alpha;

    // Temporal penalization parameter.
    // For bigger values:
    //   Solution will be driven to be similar to the previous frame solutions.
    //   To set temporal_regularization = infinity: Set to any value < 0. Solution will not change from frame to frame.
    // For smaller values:
    //   Each frame is mostly independent.
    //   Value temporal_regularization = 0: No temporal regularization, each frame is independent.
    double temporal;

    // Maximal number of iterations to perform.
    // This is only an upper bound for the maximal number of iterations. The actual number of iterations may be less than this, since the iterations will be stopped once difference between consecutive solutions is small enough.
    int iterations;

    // Determines the stopping criterion:
    // Iterations will be stopped if 1/(w*h) sum_{x,y,i} |u_{n+1}(x,y,i) - u_n(x,y,i)| < stopping_eps.
    double stop_eps;

    // The stopping criterion will be checked for only every k-th iteration, where k is given by this parameter.
    // If set to <= 0, no checking will be performed, i.e. all max_num_iterations iterations will be made.
    int stop_k;

    // If true: lambda and alpha will be adapted so that the solution will look more or less the same, for one and the same input image and for different scalings.
    //   Using this, one can run time intensive experiments on downscaled images, find suitable parameters, and then run the experiment on the original images, with the same parameters.
    //   Effectively, lambda and alpha are used as is for a "standard" image scale (640 x 480), and for a general image size w * h
    //   they are set to
    //     lambda_actual = lambda * factor
    //     alpha_actual = alpha * factor * factor
    //     where factor = sqrt(640 * 480) / sqrt(w * h).
    // If false: lambda and alpha are used as is, regardless of image scale.
    bool adapt_params;

    // If true: The regularizer will be adjust to smooth less at pixels with high edge probability
    // This changes the regularizer to
    //
    //   min(alpha * |gradient u(x, y)| ^ 2, lambda * weight(x, y)),
    //
    // instead of the original regularizer with weight(x, y) = 1 for all (x, y).
    // The weight is set as
    //
    //   weight(x, y) := exp(-|gradient_of_input_image(x, y)| / sigma),
    //
    //   with  sigma := 1 / (w * h) * sum_{pixels (x, y)} |gradient_of_input_image(x, y)|.
    //
    bool weight;

	// If true: The edge set will be overlaid on top of the computed color image.
	bool edges;

	// Compute everything with double instead of float.
    bool use_double;

    // Use CPU or CUDA.
	int engine;
	static const int engine_cpu = 0;
	static const int engine_cuda = 1;

	// If true: Output information:
	//   - image dimensions
	//   - required memory
	//   - number of iterations after which the stopping criterion has been reached
	//   - time for computation only (excluding allocation and initialization)
	//   - energy
	bool verbose;
};


// p_impl design pattern to reduce header to the minimum in order to avoid unnecessary dependencies
class SolverImplementation3;

// forward-declare here to avoid unnecessary dependencies
class ArrayDim3;
class BaseVolume;

// class instead of function to be able to maintain state, i.e. use same memory allocation for several images
class Solver3
{
public:
	Solver3();
	~Solver3();

	// general
	BaseVolume* run(const BaseVolume *in, const Par3 &par);

	// layered real
	void run(float *&out_volume, const float *in_volume, const ArrayDim3 &dim, const Par3 &par);
	void run(double *&out_volume, const double *in_volume, const ArrayDim3 &dim, const Par3 &par);

	// interlaced char
	void run(unsigned char *&out_volume, const unsigned char *in_volume, const ArrayDim3 &dim, const Par3 &par);

	// VolMat
	VolMat run(const VolMat in_image, const Par3 &par);

private:
	Solver3(const Solver3 &other_solver);  // disable
	Solver3& operator= (const Solver3 &other_solver);  // disable

	SolverImplementation3 *implementation;
};



#endif // SOLVER_H
