/**
 * @file        main.cpp
 * @author      Zongxiao He (zongxiah@bcm.edu)
 * @copyright   Copyright 2013, Leal Group
 * @date        2013-11-16
 * @see		main.h
 *
 * \brief Main function for movielens project.
 *
 * Example Usage:
 * 
 */

/*#include "main.h"*/

#include "Argument_helper.h"
#include "utilities.h"
#include "basic.h"
#include "cross_validation.h"
#include "baseline.h"
#include "neighborhood.h"
#include "svd.h"
#include "svdasym.h"
#include "svdplusplus.h"
#include "svdneighbor.h"


int main(int argc, char* argv[])
{
	// model & dataset
	std::string proj("MyProject");
	std::string model("neighbor");
	std::string training("probe");
	// step size
	double gamma0(0.001), gamma1(0.002), gamma2(0.002), gamma3(0.02);

	// regulatization parameter
	double lambda0(0.01); //lambdai(1.1), lambdau(8.0);
	double lambda6(0.008), lambda7(0.002), lambda8(0.002);

	// generic parameter
	double alpha(0.5), slow_rate(0.9);
	int base_iter(15), max_iter(250); // , kfoldcv(10);
	int nfactor(200);
	int kfold(10);

	bool include_train(true);
	bool nopredict(false);
	bool debug(false);
	bool tiny_data(false);

	dsr::Argument_helper arg;
	// required arguments
	arg.new_string("project", "\n\tProject label/id. \n\t| Will be used as output file prefix \n\t", proj);
	arg.new_string("model", "\n\tModel. \n\t| OPTION: baseline; neighbor; svd; fnm \n\t* baseline: baseline model, needed to estimate lambda_i, lambda_u; \n\t* neighbor: global neighhorhood model, see koren 2008 equation (10); \n\t* svd: SVD, see recommender system book Page 152; \n\t* svdasym: Asymmetric SVD, see koren 2008 equation(13); \n\t* svdpp: SVD++, see koren 2008 equation(15); \n\t* svdneighbor: SVD++  + global neighborhood model, see koren 2008 equation(16); \n\t", model);

	arg.new_string("training", "\n\tTriaing dataset. \n\t| OPTION: wts; probe; cv \n\t* wts: train model on whole training set and predict all missing entries (); \n\t* probe: train model on partial training set and predict probe set(); \n\t* cv: cross validation on whole training set and calculate RMSE; \n\t" , training);
	// step size: A-K
	arg.new_named_double('A', "gamma0", "<float>", "\n\tStep size gamma for BaseLine model. \n\t", gamma0);
	arg.new_named_double('B', "gamma1", "<float>", "\n\tStep size gamma for integated baseLine model. \n\t", gamma1);
	arg.new_named_double('C', "gamma2", "<float>", "\n\tStep size gamma for SVD/SVDasym/SVD++ model. \n\t", gamma2);
	arg.new_named_double('D', "gamma3", "<float>", "\n\tStep size gamma for Neighborhood model. \n\t", gamma3);

	// Regularization parameter: L-X
	arg.new_named_double('L', "lambda0", "<float>", "\n\tRegularization parameter lammbda_i in BaseLine model, see Koren 2008 first paper. \n\t", lambda0);
	/* arg.new_named_double('j', "lambdau", "<float>", "\n\tRegularization parameter lammbda_u in fast BaseLine model. \n\t", lambdau); */
	arg.new_named_double('M', "lambda6", "<float>", "\n\tRegularization parameter lammbda_6 for BaseLine model. \n\t", lambda6);
	arg.new_named_double('N', "lambda7", "<float>", "\n\tRegularization parameter lammbda_7 in SVD/SVDasym/SVD++ model. \n\t", lambda7);
	arg.new_named_double('O', "lambda8", "<float>", "\n\tRegularization parameter lammbda_8 in Neighborhood model. \n\t", lambda8);

	// generic parameter: a-k
	arg.new_named_double('a', "alpha", "<float>", "\n\tNeighborhood size regularization, see Koren 2008 equation (9). \n\t", alpha);
	arg.new_named_double('b', "slowrate", "<float>", "\n\tStep size slow rate in each iteration. \n\t", slow_rate);
	arg.new_named_int('c', "baseiter", "<int>", "\n\tNumber of iteration in integrated BaseLine model. \n\t", base_iter);
	arg.new_named_int('d', "maxiter", "<int>", "\n\tMaximun iteration in optimization. \n\t", max_iter);
	arg.new_named_int('e', "factor", "<int>", "\n\tNumber of latent factors in SVD model. \n\t", nfactor);
	arg.new_named_int('f', "kfold", "<int>", "\n\tK-flod cross validation. \n\t", kfold);

	// other parameter: l-x
	arg.new_flag('l', "include_train", "\n\tShould include \"whole training set\" in the final prediction. \n\t", include_train);	
	arg.new_flag('m', "nopredict", "\n\tIf predict [probe|missing] set.\n\t", nopredict);
	arg.new_flag('n', "debug", "\n\tDebug model: more screen output.\n\t", debug);
	arg.new_flag('o', "tiny", "\n\tUse smaller dataset in ./debug for debug purpose.\n\t", tiny_data);
	
	// program information
	std::string banner = "\n\t:-------------------------------------------------------------------------:\n\t:    STAT 640 Data Mining and Statisitical Learning: MovieLens Project    :\n\t:-------------------------------------------------------------------------:\n\t:  (c)2013 Zongxiao He & Di Fu | https://bitbucket.org/Tony_He/movielens  :\n\t:-------------------------------------------------------------------------:\n";

	arg.set_name("MovieLens");
	arg.set_description(banner.c_str());
	arg.set_version(1.0);
	arg.set_author("Zongxiao He <zh6@rice.edu>");
	time_t rawtime;
	struct tm * timeinfo;
	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	arg.set_build_date(asctime (timeinfo));

	arg.process(argc, argv);

	MovieLensModel * mymodel;

	if (model == "baseline") {
		mymodel = new BaseLine(proj, gamma0, lambda0, alpha, slow_rate, max_iter, debug);
	} else if(model == "neighbor") {
		mymodel = new Neighborhood(proj, gamma0, lambda0, base_iter, gamma1, gamma3, lambda6, lambda8, alpha, slow_rate, max_iter, debug);
	} else if(model == "svd") {
		mymodel = new SVD(proj, nfactor, gamma0, lambda0, base_iter, gamma1, gamma2, lambda6, lambda7, alpha, slow_rate, max_iter, debug);
	} else if (model == "svdasym") {
		mymodel = new SVDasym(proj, nfactor, gamma0, lambda0, base_iter, gamma1, gamma2, lambda6, lambda7, alpha, slow_rate, max_iter, debug);
	}else if(model == "svdpp") {
		mymodel = new SVDpp(proj, nfactor, gamma0, lambda0, base_iter, gamma1, gamma2, lambda6, lambda7, alpha, slow_rate, max_iter, debug);
	} else if(model == "svdneighbor") {
		mymodel = new SvdNeighbor(proj, nfactor, gamma0, lambda0, base_iter, gamma1, gamma2, gamma3, lambda6, lambda7, lambda8, alpha, slow_rate, max_iter, debug);
	} else {
		std::cout << "ERROR: the model " << model << " is not defined." << std::endl;
		exit(1);
	}

 
	std::string data_folder = tiny_data ? "./debug/" : "./data/";
	vect2RN train_data; //
	vectPN probe_data; 
	if (training == "probe") {
		load_rating(data_folder + "pts_train_rating.csv", train_data);
		load_probe(data_folder + "probe_train_rating.csv", probe_data);
	} else {
		load_rating(data_folder + "wts_train_rating.csv", train_data);
	}

	if (training == "cv") {
		KfoldCV * mycv = new KfoldCV(proj, kfold, data_folder + "pts_train_rating.csv", mymodel, debug);

		mycv->RMSE();
		mycv->write_log();

		delete mycv;

	} else if (training == "probe" || training == "wts") {

		mymodel->add_data(train_data);
		if(training == "probe")
			mymodel->add_probe(probe_data);

		mymodel->train();
	
		if(! nopredict)
			mymodel->predict(include_train);
		mymodel->write_log();

	} else {
		std::cout << "ERROR: unknow training model "  << training << std::endl;
		exit(1);
	}

	delete mymodel;
}
