/**
 * @file        neighborhood.cpp
 * @author      Zongxiao He (zongxiah@bcm.edu)
 * @copyright   Copyright 2013, Leal Group
 * @date        2013-11-16
 * @see		neighborhood.h
 *
 * \brief
 *
 * Example Usage:
 * 
 */

#include "neighborhood.h"


Neighborhood::Neighborhood(std::string projid, double g0, double l0, int bi, double g1, double g3, double l6, double l8, double a, double sr, int mi, bool debug):MovieLensModel(projid, debug){

	_gamma0 = g0; 
	_lambda0 = l0; 
    _base_iter = bi; // from baseline model
	_gamma1 = g1;
	_gamma3 = g3;
	__gamma1 = g1;
	__gamma3 = g3;
	_lambda6 = l6;
	_lambda8 = l8;
	_alpha = a;
	_slow_rate = sr;
	_max_iter = mi;

	bu_.resize(USER_NUM+1, 0);
	bi_.resize(ITEM_NUM+1, 0);
	buBase_.resize(USER_NUM+1, 0);
	biBase_.resize(ITEM_NUM+1, 0);
	buNum_.resize(USER_NUM+1, 0);
	biNum_.resize(ITEM_NUM+1, 0);

	resize_matrix(w_, ITEM_NUM+1, ITEM_NUM+1, 0.0);
	resize_matrix(c_, ITEM_NUM+1, ITEM_NUM+1, 0.0);

    mean_ = 0;    
}

void Neighborhood::init_para() {

	// initialize with baseline model
	init_base_model();

	/* for(int i = 1; i < 10+1; ++i){ */
	/* 	std::cout << buNum_[i] << " " << biNum_[i] << std::endl; */
	/* } */

	reset_matrix(w_, ITEM_NUM+1, ITEM_NUM+1, 0.0);
	reset_matrix(c_, ITEM_NUM+1, ITEM_NUM+1, 0.0);

	/* for(int i = 1; i < 10+1; ++i){ */
	/* 		for(int j = 1; j < 10+1; ++j){ */
	/* 			std::cout << w_[i][j] << " "; */
	/* 		} */
	/* 		std::cout << std::endl; */
	/* } */

}

// private function

/* See koren's "Factor in the Neighors..." for detail */

void Neighborhood::init_base_model()
{
	if(DEBUG)
		std::cout << "Neighborhood: training baseline model" << std::endl;

	// TODO: should we use same alpha/slow_rate as outside model
	BaseLine * blmodel = new BaseLine(PROJ, _gamma0, _lambda0, _alpha, _slow_rate, _base_iter, false);
	blmodel->add_data(_rNodes);
	blmodel->train();
	blmodel->get_base(mean_, bu_, bi_, buNum_, biNum_);
	copy_vector(bu_, buBase_, USER_NUM+1);
	copy_vector(bi_, biBase_, ITEM_NUM+1);

	delete blmodel;
}


void Neighborhood::log_parameter(){

	Json::Value mode;
	mode["model"] = "neighbor";
	mode["probe"] = PROBE;
	mode["debug"] = DEBUG;
	mode["tinydata"] = SMALLDATA;

	Json::Value para;
	para["gamma0"] = _gamma0;
	para["gamma1"] = _gamma1; // original gammas
	para["gamma3"] = _gamma3;
	para["lambda0"] = _lambda0;
	para["lambda6"] = _lambda6;
	para["lambda8"] = _lambda8;
	para["alpha"] = _alpha;
	para["slowrate"] = _slow_rate;
	para["baseiter"] = _base_iter;
	para["maxiter"] = _max_iter;
	para["probe_iters"] = probe_iter_;
	
	Json::Value tm;
	tm["trmse"] = rmse_;
	tm["prmse"] = prmse_;

	_root["train"] = mode;
	_root["parameter"] = para;
	_root["rmse"] = tm;
}


void Neighborhood::train()
{
	__gamma1 = _gamma1;
	__gamma3 = _gamma3;

	PRINT("Neighborhood: Initializing ... ");
	init_para();
	/* mean_ = get_rating_mean(); */
	/* init_base_model(); //TODO: initialize w/c matrix */

	double last_prmse = 10000.0;
	int check_step = (SMALLDATA)?1000:100000;

	PRINT("Neighborhood: Training ... ");
	vectorF bu, bi;
	vector2F w, c;

	for(int iter = 1; iter <= _max_iter; ++iter){  
		long double rmse = 0.0;
		int n = 0;

		// hold previous parameter in case probe rmse increase!
		if (PROBE) {
			copy_vector(bu_, bu, USER_NUM+1);
			copy_vector(bi_, bi, ITEM_NUM+1);
			copy_matrix(w_, w, ITEM_NUM+1, ITEM_NUM+1);
			copy_matrix(c_, c, ITEM_NUM+1, ITEM_NUM+1);
		}

		for(int u = 1; u < USER_NUM+1; ++u) {   //process every user      
			int RuNum = _rNodes[u].size(); //process every item rated by user u 
			float sqrtRuNum = (RuNum>1) ? (1.0/pow(RuNum, _alpha)) : 0.0;
            
			//process every item rated by user u
			for(int i=0; i < RuNum; ++i) {
				RateNode rn = _rNodes[u][i];
				int iid = rn.item;
				short rui = rn.rate; 
				float pui = predict_rate(u, iid);  //predict rate
                    
				float eui = rui - pui;
                    
				if( isnan(eui) ) { 
					std::cout<<u<<"\t"<<i<<"\t"<<pui<<"\t"<<rui<<"\t"<<bu_[u]<<"\t"<<bi_[iid]<<"\t"<<mean_<<std::endl;
					exit(1);
				}

				rmse += eui * eui; ++n;
				if( DEBUG && n % check_step == 0) 
					std::cout<<"iteration "<< iter << ":\t" << n << " dealed!"<<std::endl;
                    
				bu_[u] += __gamma1 * (eui - _lambda6 * bu_[u]);
				bi_[iid] += __gamma1 * (eui - _lambda6 * bi_[iid]);
                    
				for(int j=0; j < RuNum; ++j) {
					RateNode uj = _rNodes[u][j];
					int jid = uj.item;
					double ruj = (double) uj.rate;
					float rb_uj = ruj - (mean_ + buBase_[u] + biBase_[jid]); // fixed bias

					w_[iid][jid] +=  __gamma3 * (sqrtRuNum*eui*rb_uj - _lambda8*w_[iid][jid]);
					c_[iid][jid] +=  __gamma3 * (sqrtRuNum*eui - _lambda8*c_[iid][jid]);
				}
            }
		}

		rmse_ =  sqrt( rmse / n);
		if(DEBUG)
			std::cout << "Neighborhood: iteration " << iter << " RMSE " << rmse_ << std::endl;
        
		if (PROBE){
			prmse_ = Probe_RMSE();
			if(DEBUG) std::cout << "Neighborhood: iteration " << iter << " Probe RMSE " << last_prmse << " -> " << prmse_ << std::endl;

			if( std::abs(prmse_ - last_prmse) < 1.0e-5 && iter >= 5) {
				probe_iter_ = iter;
				copy_vector(bu, bu_, USER_NUM+1);
				copy_vector(bi, bi_, ITEM_NUM+1);
				copy_matrix(w, w_, ITEM_NUM+1, ITEM_NUM+1);
				copy_matrix(c, c_, ITEM_NUM+1, ITEM_NUM+1);

				/* std::cout << "matrix w "; mat_mean_std(w_, ITEM_NUM+1, ITEM_NUM+1);  */
				/* std::cout << "matrix c "; mat_mean_std(c_, ITEM_NUM+1, ITEM_NUM+1); */
				break; 
			} else
				last_prmse = prmse_;
		}

		__gamma1 *= _slow_rate;
		__gamma3 *= _slow_rate;
	}

	log_parameter();
	PRINT("Neighborhood: Training done ");
	return;
}


float Neighborhood::predict_rate(int uid, int iid){
	std::vector<RateNode> Ru = _rNodes[uid];
    int RuNum = Ru.size(); 
    double ret = mean_ + bu_[uid] + bi_[iid]; 

	/* std::cout << ret << std::endl; */

    if(RuNum > 1){
		double sumEx(0.0), sumIm(0.0);
		float sqrtRuNum = 1/pow(RuNum, _alpha);
		double mbu = mean_ + buBase_[uid];

		/* std::cout <<":mbu " <<  buBase_[uid] << "\n"; */
		for(int j=0; j < RuNum; ++j) {
			RateNode uj = Ru[j];
			int jid = uj.item;
			double rate = (double) uj.rate;
			/* std::cout << jid << " " << rate << " " << w_[iid][jid] << std::endl; */
			/* std::cout << sumIm << std::endl; */
			sumEx += (rate - mbu - biBase_[jid]) * w_[iid][jid];
			sumIm += c_[iid][jid];
		}
        
		ret += sqrtRuNum * (sumEx+sumIm);
	}

	if( isnan(ret) ) { 
		std::cout<<uid<<"\t"<<iid<<"\t"<<mean_<<"\t"<<bu_[uid]<<"\t"<<bi_[iid]<<"\t"<<RuNum<<std::endl;
		exit(1);
	}

    if(ret <= 1.0) ret = 1;
    if(ret >= 5.0) ret = 5;
    return ret;
}
