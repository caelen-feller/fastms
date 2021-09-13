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

#include "example_volumes.h"

#include "solver/volume_solver.h"
#include "param.h"
#include "volume_util.h"
#include "util/volume_mat.h"


#include <cmath>
#include <vector>
#include <iostream>
#include <ctime>
#include <algorithm>


int example_volumes(int argc, char **argv)
{
    bool show_help = false;
	get_param("help", show_help, argc, argv);
	get_param("-help", show_help, argc, argv);
	get_param("h", show_help, argc, argv);
	get_param("-h", show_help, argc, argv);
    if (show_help) { std::cout << "Usage: " << argv[0] << " -i <inputfiles>" << std::endl; return 0; }

    // get params
    Par3 par;
    get_param("verbose", par.verbose, argc, argv);
    if (par.verbose) std::cout << std::boolalpha;
    get_param("lambda", par.lambda, argc, argv);
    get_param("alpha", par.alpha, argc, argv);
    get_param("temporal", par.temporal, argc, argv);
    get_param("iterations", par.iterations, argc, argv);
    get_param("stop_eps", par.stop_eps, argc, argv);
    get_param("stop_k", par.stop_k, argc, argv);
    get_param("adapt_params", par.adapt_params, argc, argv);
    get_param("weight", par.weight, argc, argv);
    get_param("use_double", par.use_double, argc, argv);
    {
    	std::string s_engine = "";
        if (get_param("engine", s_engine, argc, argv))
        {
        	std::transform(s_engine.begin(), s_engine.end(), s_engine.begin(), ::tolower);
        	if (s_engine.find("cpu") == 0 || s_engine.find("host") == 0)
        	{
        		par.engine = Par::engine_cpu;
        	}
        	else if (s_engine.find("cuda") == 0 || s_engine.find("device") == 0)
        	{
        		par.engine = Par::engine_cuda;
        	}
        	else
        	{
        		get_param("engine", par.engine, argc, argv);
        	}
        }
    }
    get_param("edges", par.edges, argc, argv);
    if (par.verbose) { par.print(); }
    std::cout << std::endl;

    int slice2d = -1;
    get_param("slice2d", slice2d, argc, argv);
    if (par.verbose) std::cout << "  slice2d: "; 
    if (slice2d == -1) std::cout << "-1 (processing as 2d image)" << std::endl; 
    else std::cout << "processing only slice " << slice2d << " as 2d image" << std::endl; 

    bool show_result = true;
    get_param("show", show_result, argc, argv);
    if (show_result) std::cout << "Result showing is unimplemented at this stage" << std::endl;

    std::string save_dir = "volume_output";
    get_param("save", save_dir, argc, argv);
    bool save_result = (save_dir != "");
    if (par.verbose && save_result) { std::cout << "  save (RESULTS DIRECTORY): " << save_dir.c_str() << std::endl; }
    else { std::cout << "  save (results directory): empty (result #else
	std::cerr << "ERROR: " << __FILE__ << ": OpenCV disabled in compilation, but this example requires OpenCV." << std::endl;
#endif // DISABLE_OPENCV
saving disabled))" << std::endl; }

    std::cout << std::endl;

    std::vector<std::string> input_names;
    // TODO: Support for image stacks (2d vec of opencv images)
    std::vector<MatVolume> input_volumes;

    std::vector<std::string> inputfiles;
    bool has_i_param = get_param("i", inputfiles, argc, argv);
    if (!has_i_param)
    {
    	std::string default_file = "volumes/sphere.data";
    	//std::cerr << "Using " << default_file << " (no option \"-i <inputfiles>\" given)" << std::endl;
    	inputfiles.push_back(default_file);
    }
    
	if (par.verbose) std::cout << "loading input files" << std::endl;
	for (int i = 0; i < (int)inputfiles.size(); i++)
	{
		MatVolume input_volume = volread(inputfiles[i].c_str());
		if (input_volume.mat.data == NULL) { std::cerr << "ERROR: Could not load volume " << inputfiles[i].c_str() << std::endl; continue; }
		input_volumes.push_back(input_volume);
		input_names.push_back(inputfiles[i]);
	}
    if (input_volumes.size() == 0)
    {
    	std::cerr << "No input files" << std::endl;
    	return -1;
    }

    // for 1d processing: extract rows
    if (slice2d >= 0)
    {
        for (int i = 0; i < (int)input_volumes.size(); i++)
        {
        	input_volumes[i] = extract_row(input_volumes[i], slice2d);
        }
    }

    // process
    std::vector<cv::Mat> result_volumes(input_volumes.size());
    Solver solver;
    for (int i = 0; i < (int)input_volumes.size(); i++)
    {
    	if (par.verbose) std::cout << input_names[i].c_str() << ":  ";
    	result_volumes[i] = solver.run(input_volumes[i], par);
    }

    // for 1d processing: replace 1d input and result with its graph visualization
    if (slice2d >= 0)
    {
    	int graph_height = 200;
    	double thresh_jump = (par.alpha > 0.0? std::sqrt(par.lambda / par.alpha) : -1.0);
        for (int i = 0; i < (int)result_volumes.size(); i++)
        {
        	input_volumes[i] = volume1d_to_graph(input_volumes[i], graph_height, -1.0);
        	result_volumes[i] = volume1d_to_graph(result_volumes[i], graph_height, thresh_jump);
        }
    }


    // show results
    if (save_result)
    {
        for (int i = 0; i < (int)input_volumes.size(); i++)
        {
        	std::string dir;
			std::string basename;
			FilesUtil::to_dir_basename(input_names[i], dir, basename);
			if (slice2d >= 0) { std::stringstream s; s << "_row" << slice2d; basename += s.str(); }

			std::string out_dir = save_dir + '/' + dir;
			if (!FilesUtil::mkdir(out_dir)) { std::cerr << "ERROR: Could not create output directory " << out_dir.c_str() << std::endl; continue; }
			std::string out_file_input = out_dir + '/' + basename + "__input.png";
			std::string out_file_result = out_dir + '/' + basename + "__result" + par_to_string(par) + ".png";
			if (!cv::imwrite(out_file_input, input_volumes[i])) { std::cerr << "ERROR: Could not save input volume " << out_file_input.c_str() << std::endl; continue; }
			if (!cv::imwrite(out_file_result, result_volumes[i])) { std::cerr << "ERROR: Could not save result volume " << out_file_result.c_str() << std::endl; continue; }
			std::cout << "SAVED RESULT: " << out_file_result.c_str() << "  (SAVED INPUT: " << out_file_input.c_str() << ")" << std::endl;
        }
    }
    if (show_result)
    {
        for (int i = 0; i < (int)input_volumes.size(); i++)
        {
        	show_volume("Input", input_volumes[i], 100, 100);
        	show_volume("Output", result_volumes[i], 100 + input_volumes[i].cols + 40, 100);
        	cv::waitKey(0);
        }
    }
	cv::destroyAllWindows();

	return 0;
}
