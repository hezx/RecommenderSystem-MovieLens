/**
 * @file        cross_validation.cpp
 * @author      Zongxiao He (zongxiah@bcm.edu)
 * @copyright   Copyright 2013, Leal Group
 * @date        2013-11-18
 * @see		cross_validation.h
 *
 * \brief
 *
 * Example Usage:
 * 
 */

#include "cross_validation.h"

KfoldCV::KfoldCV(std::string proj, int k, std::string data_file, MovieLensModel* model, bool debug){

	_proj = proj;
	K = k;
	_file = data_file;
	_model = model;
	DEBUG = debug;

	if(DEBUG)
		std::cout << "Cross-Validation ..." << std::endl;

	read_rating_kfold();
}


void KfoldCV::read_rating_kfold()
{
	std::ifstream file;
	file.open(_file.c_str());
    if (!file.is_open()) {
		std::cout << "ERROR: can't open input file " + _file +"\n";
        exit(1);
    }
	// remove header
	std::string header; std::getline(file, header);

    _cvNodes.resize(USER_NUM+1); 

    std::string line;
	while(std::getline(file, line)){
		std::stringstream ss(line);
		std::string idstr; // ID
		std::getline(ss, idstr, ',');
		std::string rate; // 
		std::getline(ss, rate, ',');
		
		int id = atoi(idstr.c_str()); // id = (itemid - 1)*6040 + userid
		short userid = id % USER_NUM;
		if (userid == 0) userid = USER_NUM;
		short itemid = id / USER_NUM + 1;
		
		
		CVRateNode tmp(itemid, (short) atoi(rate.c_str()), assign_group());
		_cvNodes[userid].push_back(tmp);
	}
	return;
}


short KfoldCV::assign_group()
{
	double step = 1.0 / K;
	double r = ((double) rand() / RAND_MAX);
	return (short) ceil(r/step);
}


void KfoldCV::get_k_group(int k, vect2RN& train, vectPN& test)
{
	train.resize(USER_NUM+1);
	test.resize(0);

	for(int u = 1; u < USER_NUM+1; ++u) {
		train[u].resize(0);

		int RuNum = _cvNodes[u].size(); 
		for(int i=0; i < RuNum; ++i) {
			CVRateNode rn = _cvNodes[u][i];
			if (rn.group == k) {
				ProbeNode tmp((short) u, rn.item, rn.rate);
				test.push_back(tmp);
			} else {
				RateNode tmp(rn.item, rn.rate);
				train[u].push_back(tmp);
			}
		}
	}
}


void KfoldCV::RMSE(){
	vectorF rmses;
	vect2RN train;
	vectPN test;
	
	for(int i=1; i<=K; i++){
		get_k_group(i, train, test);
		_model->add_data(train);
		_model->train();
		double rmse = _model->evaluate(test);
		rmses.push_back(rmse);

		if(DEBUG) 
			std::cout << "Cross-Validation fold " << i << " RMSE " << rmse << std::endl;

		if(i == 1)
			_cvroot = _model->get_log();
	}

	_cv_res = mean_std(rmses);

	if(DEBUG) 
		std::cout << "Cross-Validation done! " << " mean RMSE " << _cv_res[0] << " +/-" << _cv_res[1] << std::endl;
	return;
}



void KfoldCV::write_log(){
	
	Json::Value cv;
	cv["kfold"] = K;
	cv["rmse_mean"] = _cv_res[0];
	cv["rmse_std"] = _cv_res[1];
	_cvroot["cv"] = cv;
	_cvroot["project"] = _proj;

	std::string logname = "./LogFiles/" + _proj  + ".json";
	if(DEBUG)
		std::cout << "Cross-Validation: writing parameters into log file " << logname << std::endl;

	std::ofstream log;
	log.open(logname.c_str());

	if(!log.is_open()){
		std::cout << "ERROR: can't open out file " + logname +"\n";
        exit(1);
    }

	Json::StyledStreamWriter writer;
	writer.write(log, _cvroot);
	log.close();

	return;
}
