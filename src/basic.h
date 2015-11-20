/**
 * \file        base_class.h
 * \author      Zongxiao He (zongxiah@bcm.edu)
 * \copyright   Copyright 2013, Leal Group
 * \date        2013-11-16
 *
 * \brief
 *
 *
 */


#if SMALLDATA
    #define USER_NUM 600
    #define ITEM_NUM 300
#else
    #define USER_NUM 6040
    #define ITEM_NUM 3187
#endif

#define CUTOFF 5.0e-6

#ifndef BASE_CLASS_H
#define BASE_CLASS_H

#include "utilities.h"


struct RateNode
{
	RateNode(){};
RateNode(short i, short r):item(i), rate(r){}
    short item;
    short rate;
};

struct ProbeNode
{
	ProbeNode(){}
ProbeNode(short u, short i, short r):user(u), item(i), rate(r){}
    short user;
    short item;
    short rate;
};


typedef std::vector< std::vector<RateNode> > vect2RN;
typedef std::vector<ProbeNode> vectPN;



class MovieLensModel{
public:
MovieLensModel(std::string projid, bool debug):PROJ(projid), DEBUG(debug), PROBE(false){}

	virtual void train() = 0;

	void predict(bool include_train);

	void add_data(vect2RN& rate){
		_rNodes = rate;
	}
	void add_probe(vectPN& probe_set){
		PROBE = true; 
		_pNodes = probe_set;
	}

	double evaluate(vectPN& pnodes);

	void write_log();

	Json::Value get_log(){
		return _root;
	}

	virtual ~MovieLensModel(){};

protected:
	std::string PROJ;
	Json::Value _root;
	// flag
	bool DEBUG, PROBE;
	// input 
	vect2RN _rNodes;
	vectPN _pNodes;

	virtual float predict_rate(int user, int item) = 0; 

	double get_rating_mean();
	double Probe_RMSE();

	void PRINT(std::string msg){
		if(DEBUG)
			std::cout << msg << std::endl;
		return;
	}

	std::map<short, short> iid2rate(std::vector<RateNode> & rates);
};


// Other functions
void load_rating(std::string filename, vect2RN& rMatrix);
void load_probe(std::string filename, vectPN& pMatrix);

vectorF mean_std(vectorF & vals);
void mat_mean_std(vector2F & mat, int nrow, int ncol);
void set_random_matrix(vector2F & vect, int row, int col);
#endif /* BASE_CLASS_H */
