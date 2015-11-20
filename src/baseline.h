/**
 * \file        baseline.h
 * \author      Zongxiao He (zongxiah@bcm.edu)
 * \copyright   Copyright 2013, Leal Group
 * \date        2013-11-16
 *
 * \brief Part of the code is borrowed from https://code.google.com/p/recsyscode/
 *
 *
 */


#ifndef BASELINE_H
#define BASELINE_H

#include "utilities.h"
#include "basic.h"


class BaseLine: public MovieLensModel{

public:
	BaseLine(std::string projid, double g0, double l0, double a, double sr, int mi, bool debug);
	void train();
	void predict();
	
	void get_base(double & gm, vectorF & bu, vectorF & bi, vectorUI & buNum, vectorUI & biNum){
		gm = mean_;
		copy_vector(bu_, bu, USER_NUM+1);
		copy_vector(bi_, bi, ITEM_NUM+1);
		copy_vector(buNum_, buNum, USER_NUM+1);
		copy_vector(biNum_, biNum, ITEM_NUM+1);
	}

	~BaseLine(){}

private:
	// training paramter
	double _lambdau, _lambdai; // from baseline model
	double _gamma0, _lambda0, _alpha, _slow_rate;
	int _max_iter;

	// model parameter
	double __gamma0;
	vectorF bu_, bi_;  // the user and movie bias
    vectorUI buNum_, biNum_;       // num of ratings given by user, num of ratings given to item

    float mean_;      // mean of all ratings
	int probe_iter_;
	double rmse_, prmse_;

	float predict_rate(int user, int item);

	void log_parameter();

	void init_base_model();
	void init_para();
};

#endif /* NEIGHBORHOOD_H */
