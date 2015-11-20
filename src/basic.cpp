/**
 * @file        basic.cpp
 * @author      Zongxiao He (zongxiah@bcm.edu)
 * @copyright   Copyright 2013, Leal Group
 * @date        2013-11-16
 * @see		basic.h
 *
 * \brief
 *
 * Example Usage:
 * 
 */

#include "basic.h"


double MovieLensModel::get_rating_mean()
{
    double sum = 0;
    int num = 0;
    for(unsigned int i = 1; i < USER_NUM+1; ++i){
        for(unsigned int j=0; j < _rNodes[i].size(); ++j) {
            sum += _rNodes[i][j].rate;
            ++num;
        }
    }
    return sum/num;
}


double MovieLensModel::evaluate(vectPN& pnodes){

	long double prmse = 0;
	int psize = pnodes.size();

	double prate, err;
	for (int i=0; i<psize; ++i){
		ProbeNode pn = pnodes[i];
		prate = predict_rate(pn.user, pn.item);
		err = pn.rate - prate;
		prmse += err*err;
	 }
	
	return (psize == 0)? 0 : sqrt(prmse/psize);
}


double MovieLensModel::Probe_RMSE(){
	if (PROBE == false) {
		std::cout << "ERROR: no probe set loaded before using Probe_RMSE" << std::endl;
		exit(1);
	}
	
	return evaluate(_pNodes);
}


void MovieLensModel::write_log(){
	
	std::string logname = "./LogFiles/" + PROJ  + ".json";
	PRINT("MovieLens: writing parameters into log file " + logname);

	std::ofstream log;
	log.open(logname.c_str());

	if(!log.is_open()){
		std::cout << "ERROR: can't open out file " + logname +"\n";
        exit(1);
    }

	_root["project"] = PROJ;

	Json::StyledStreamWriter writer;
	writer.write(log, _root);
	log.close();

	return;
}



void MovieLensModel::predict(bool include_train){

	std::ofstream pfile;
	std::string pout = "./PredictRating/" + PROJ  + ".csv";
	pfile.open(pout);
	pfile << "ID,Rating" << std::endl;

	double prate;
	int id, n = 0;
	int check_step = (SMALLDATA)?10000:1000000;

	// predict probe set if PROBE
	if(PROBE){
		PRINT("MovieLens: predict all probe set into " + pout);

		int psize = _pNodes.size();

		for (int i=0; i<psize; ++i){
			ProbeNode pn = _pNodes[i];
			prate = predict_rate(pn.user, pn.item);
			id = (pn.item - 1)*USER_NUM + pn.user;
			pfile << id << "," << prate << std::endl;
		}
	} else {

		// replace training entry with exact rate
		std::string wo = (include_train)? "with" : "without";
		PRINT("MovieLens: predict entries " + wo + " original training rate into " + pout);

		for(int u=1; u <= USER_NUM; u++){
			// build dict itemid-> rate
			std::map<short, short> irmap = iid2rate(_rNodes[u]);

			for(int i=1; i <= ITEM_NUM; i++) {
				id = (i - 1)*USER_NUM + u;
				std::map<short, short>::iterator it = irmap.find(i);

				// found, then replace
				if(it != irmap.end()) {
					if (include_train) {
					prate = it->second;
					pfile << id << "," << prate << std::endl;
					}
				} else {
					prate = predict_rate(u, i);
					pfile << id << "," << prate << std::endl;
				}
					
				++n;
				if( DEBUG && n % check_step == 0) 
					std::cout << "\t" << n << " dealed!"<<std::endl;

			}
		}
	}
	
	pfile.close();
	return;
}


std::map<short, short> MovieLensModel::iid2rate(std::vector<RateNode> & rates){
	std::map<short, short> themap;
	for(int j=0; j < rates.size(); j++){
		RateNode jrate = rates[j];
		themap[jrate.item] = jrate.rate;
	}
	return themap;
}



void load_rating(std::string filename, vect2RN& rMatrix)
{
	std::ifstream file;
	file.open(filename.c_str());
    if (!file.is_open()) {
		std::cout << "ERROR: can't open input file "  << filename << std::endl;
        exit(1);
    }
	// remove header
	std::string header; std::getline(file, header);

    rMatrix.resize(USER_NUM+1); 

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
		short itemid = (id - 1) / USER_NUM + 1; // Update: (id - 1) Mon Nov 25 13:50:38 CST 2013

		
		// std::cout << filename << " " << USER_NUM << " " << ITEM_NUM << " " << id << " " <<userid << " " << itemid << " " << rate << std::endl;
		RateNode tmp(itemid, (short) atoi(rate.c_str()));
		rMatrix[userid].push_back(tmp);
	}
	return;
}


void load_probe(std::string filename, vectPN& pMatrix)
{
	std::ifstream file;
	file.open(filename.c_str());
    if (!file.is_open()) {
		std::cout << "ERROR: can't open input file " + filename +"\n";
        exit(1);
    }

	// remove header
	std::string header; std::getline(file, header);

    pMatrix.resize(0); 

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
		short itemid = (id - 1) / USER_NUM + 1;
		
		ProbeNode tmp(userid, itemid, (short) atoi(rate.c_str()));
		pMatrix.push_back(tmp);
	}
	return;
}


vectorF mean_std(vectorF & vals)
{
	double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
	double mean = sum / vals.size();

	double sq_sum = std::inner_product(vals.begin(), vals.end(), vals.begin(), 0.0);
	double stdev = std::sqrt(sq_sum / vals.size() - mean * mean);

	vectorF res;
	res.push_back(mean); res.push_back(stdev);
	return res;
};



void mat_mean_std(vector2F & mat, int nrow, int ncol)
{
	vectorF items;
	for(int i = 1; i < nrow; ++i) {
		for(int j=1; j < ncol; ++j) {
			items.push_back(mat[i][j]);
		}
	}
	vectorF res = mean_std(items);
	std::cout << res[0] << " " << res[1] << std::endl;
	return;
}


/**
 * see https://code.google.com/p/recsyscode/
 */
void set_random_matrix(vector2F & vect, int row, int col)
{
	reset_matrix(vect, row, col, 0.0);

	for(int i=0; i<row; ++i){
		for(int j=0; j<col; ++j) {
			double r = 0.01 * (rand()/(double)RAND_MAX) / sqrt(col);
			vect[i][j] = r;
		}
	}
	return;
}
