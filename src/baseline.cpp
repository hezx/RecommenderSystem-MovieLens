/**
 * @file        baseline.cpp
 * @author      Zongxiao He (zongxiah@bcm.edu)
 * @copyright   Copyright 2013, Leal Group
 * @date        2013-11-16
 * @see		baseline.h
 *
 * \brief
 *
 * Example Usage:
 * 
 */

#include "baseline.h"


BaseLine::BaseLine(std::string projid, double g0, double l0, double a, double sr, int mi, bool debug):MovieLensModel(projid, debug){

	_gamma0 = g0;
	_lambda0 = l0;
	_alpha = a;
	_slow_rate = sr;
	_max_iter = mi;

	bu_.resize(USER_NUM+1, 0);
	bi_.resize(ITEM_NUM+1, 0);
	buNum_.resize(USER_NUM+1, 0);
	biNum_.resize(ITEM_NUM+1, 0);

}

// private function


void BaseLine::init_para() {

	std::fill(bu_.begin(), bu_.end(), 0);
	std::fill(bi_.begin(), bi_.end(), 0);
	std::fill(buNum_.begin(), buNum_.end(), 0);
	std::fill(biNum_.begin(), biNum_.end(), 0);

    mean_ = 0;    
}


/* See koren's "Factor in the Neighors..." for detail */
void BaseLine::log_parameter(){

	Json::Value mode;
	mode["model"] = "baseline";
	mode["probe"] = PROBE;
	mode["debug"] = DEBUG;
	mode["tinydata"] = SMALLDATA;

	Json::Value para;
	para["gamma0"] = _gamma0; // original gammas
	para["lambda0"] = _lambda0;
	para["alpha"] = _alpha;
	para["slowrate"] = _slow_rate;
	para["maxiter"] = _max_iter;
	para["probe_iters"] = probe_iter_;

	Json::Value tm;
	tm["trmse"] = rmse_;
	tm["prmse"] = prmse_;

	_root["train"] = mode;
	_root["parameter"] = para;
	_root["rmse"] = tm;
}


void BaseLine::init_base_model()
{
	for(int i = 1; i < USER_NUM+1; ++i){
		int ncol = _rNodes[i].size();

		for(int j=0; j < ncol; ++j) {
			RateNode rn = _rNodes[i][j];
			bi_[rn.item] += (rn.rate - mean_);
			biNum_[rn.item] += 1;
		}            
	}
 
	for(int i = 1; i < ITEM_NUM+1; ++i) {
		if(biNum_[i] >= 1) 
			bi_[i] = bi_[i]/(biNum_[i]);
		else 
			bi_[i] = 0.0;
	}
       
	// update bu
	for(int i = 1; i < USER_NUM+1; ++i){
		int ncol = _rNodes[i].size();

		for(int j=0; j < ncol; ++j) {
			RateNode rn = _rNodes[i][j];
			bu_[i] += (rn.rate - mean_ - bi_[rn.item]);
			buNum_[i] += 1;
		}            
	}
	for(int i = 1; i < USER_NUM+1; ++i) {
		if(buNum_[i]>=1)
			bu_[i] = bu_[i]/(buNum_[i]);
		else 
			bu_[i] = 0.0;
	}
}


void BaseLine::train()
{
	init_para();
	__gamma0 = _gamma0;

	PRINT("BaseLine: Training ... ");
	vectorF bu, bi;

	mean_ = get_rating_mean();
	init_base_model();

	double last_prmse = 10000.0;
	int check_step = (SMALLDATA)?1000:100000;

	for(int iter = 1; iter <= _max_iter; ++iter){  

		long double rmse = 0.0;
		int n = 0;

		// hold previous parameter in case probe rmse increase!
		if (PROBE) {
			copy_vector(bu_, bu, USER_NUM+1);
			copy_vector(bi_, bi, ITEM_NUM+1);
		}

		for(int u = 1; u < USER_NUM+1; ++u) {   //process every user      
			int RuNum = _rNodes[u].size(); //process every item rated by user u 
            
			for(int i=0; i < RuNum; ++i) {
				RateNode rn = _rNodes[u][i];
				int iid = rn.item;
				short rui = rn.rate; 
				float pui = predict_rate(u, iid);  //predict rate
                    
				float eui = rui - pui;
				rmse += eui * eui; ++n;

				if( DEBUG && n % check_step == 0) 
					std::cout<<"iteration "<< iter << ":\t" << n << " dealed!"<<std::endl;

				bu_[u] += __gamma0 * (eui - _lambda0 * bu_[u]);
				bi_[iid] += __gamma0 * (eui - _lambda0 * bi_[iid]);
			}
		}

		rmse_ =  sqrt( rmse / n);
		if(DEBUG)
			std::cout << "Neighborhood: iteration " << iter << " RMSE " << rmse_ << std::endl;
        
		if (PROBE){
			prmse_ = Probe_RMSE();
			if(DEBUG)
				std::cout << "Neighborhood: Probe RMSE " << prmse_ << std::endl;

			if( std::abs(prmse_ - last_prmse) < 1.0e-5 && iter >= 5) {
				probe_iter_ = iter;
				copy_vector(bu, bu_, USER_NUM+1);
				copy_vector(bi, bi_, ITEM_NUM+1);
				break; 
			} else
				last_prmse = prmse_;
		} 
	}

	__gamma0 *= _slow_rate;
	log_parameter();

	PRINT("BaseLine: Training done ");

	return;
}


float BaseLine::predict_rate(int uid, int iid){

    double ret = mean_ + bu_[uid] + bi_[iid]; 
    if(ret <= 1.0) ret = 1;
    if(ret >= 5.0) ret = 5;
    return ret;
}
