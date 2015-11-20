/**
 * \file        neighborhood.h
 * \author      Zongxiao He (zongxiah@bcm.edu)
 * \copyright   Copyright 2013, Leal Group
 * \date        2013-11-16
 *
 * \brief Part of the code is borrowed from https://code.google.com/p/recsyscode/
 *
 *
 */


#ifndef NEIGHBORHOOD_H
#define NEIGHBORHOOD_H

#include "utilities.h"
#include "basic.h"
#include "baseline.h"

class Neighborhood: public MovieLensModel{

public:
	Neighborhood(std::string projid, double g0, double l0, int bi, double g1, double g2, double l6, double l8,  double a, double sr, int mi, bool debug);
	void train();
	void predict();
	~Neighborhood(){}

private:
	// training paramter
	double _gamma0, _lambda0; // from baseline model
	int _base_iter;
	double _gamma1, _gamma3, _lambda6, _lambda8, _alpha, _slow_rate;
	int _max_iter;

	// model parameter
	double __gamma1, __gamma3;
	vectorF bu_, bi_;  // the user and movie bias
    vectorF buBase_, biBase_;    //stored and unchanged bias of user and item
    vectorUI buNum_, biNum_;       // num of ratings given by user, num of ratings given to item
    
	vector2F w_, c_; // weight of explicit and implicit
    /* float w_[ITEM_NUM+1][ITEM_NUM+1];   // weight of explicit */
    /* float c_[ITEM_NUM+1][ITEM_NUM+1];   // weight of implicit */
    double mean_;                         // mean of all ratings

	int probe_iter_;
	double rmse_, prmse_;

	float predict_rate(int user, int item);

	void init_base_model();
	void init_para();
	void log_parameter();

};

#endif /* NEIGHBORHOOD_H */
