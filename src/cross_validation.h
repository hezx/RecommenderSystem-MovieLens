/**
 * \file        cross_validation.h
 * \author      Zongxiao He (zongxiah@bcm.edu)
 * \copyright   Copyright 2013, Leal Group
 * \date        2013-11-18
 *
 * \brief
 *
 *
 */


#ifndef CROSS_VALIDATION_H
#define CROSS_VALIDATION_H

#include "utilities.h"
#include "basic.h"

struct CVRateNode
{
CVRateNode(short i, short r, short g):item(i), rate(r), group(g){}
    short item;
    short rate;
	short group;
};

typedef std::vector< std::vector<CVRateNode> > vect2CVRN;


class KfoldCV {

public:
	KfoldCV(std::string proj, int k, std::string data_file, MovieLensModel* model, bool debug);
	void RMSE();
	void write_log();

private:
	std::string _proj;
	int K;
	std::string _file;
	MovieLensModel* _model;
	bool DEBUG;

	vect2CVRN _cvNodes;
	Json::Value _cvroot;
	vectorF _cv_res;

	void read_rating_kfold();

	short assign_group();

	void get_k_group(int k, vect2RN& train, vectPN& test);
};



#endif /* CROSS_VALIDATION_H */
