//
//  main.cpp
//  Estimate probit using NLopt
//  http://ab-initio.mit.edu/wiki/index.php/NLopt_Reference#The_nlopt_opt_object
//
#define _USE_MATH_DEFINES


#include <iostream>
#include <vector>
#include <cstdio>
#include <assert.h>
#include <fstream>
#include <cmath>

#include "nlopt.hpp"

using namespace std;


// create class that contains data
class data_estimation {

public:
	// define vectors of variables in the data
	int numobs;
	vector<int> obsid, educ, n, dec;
	vector<double> latentvar, wage;
};

// Function that calculates a normal CDF
double normalCDF(double value, double mu, double sigma);
double pdf(double x, double sigma);


//The objective function
double objfunc(const std::vector<double> &x, std::vector<double> &grad, void*Domain);

int main() {

	const int nobs = 1000;
	const unsigned int nparam = 6;

	// declare mydata variable that will contain our data
	data_estimation mydata;

	//Set the size of the vectors to the size of our dataset
	mydata.numobs = nobs;
	mydata.obsid.resize(nobs, 0);
	mydata.educ.resize(nobs, 0);
	mydata.n.resize(nobs, 0);
	mydata.dec.resize(nobs, 0);
	mydata.latentvar.resize(nobs, 0.0);
	mydata.wage.resize(nobs, 0.0);

	printf("Reading in data\n");

	// Create ifstream to read data from file
	ifstream infile;
	infile.open("D:/Cloud Filing/Dropbox/My Files/ASU Courses/Fall-2016/Greg-Nora/Greg/Static Discrete Choice/Assignment/SimulateModel2/sim_data_assignment.txt");

	// loop over observations and read in data
	for (int iobs = 0; iobs < nobs; iobs++) {

		//Read in four numbers
		infile >> mydata.obsid.at(iobs) >> mydata.educ.at(iobs) >> mydata.n.at(iobs) >> mydata.latentvar.at(iobs) >> mydata.dec.at(iobs) >> mydata.wage.at(iobs);

		//Check that we haven't gotten to the end of the file before we expected
		if (!infile.good()) {
			cout << "***ABORTING: Problem reading in data\n";
			assert(0);
		}
		// Print out the data to make sure it looks right
		//  printf("%5d %10d %16.10f %10d\n",mydata.obsid.at(iobs),mydata.educ.at(iobs),mydata.latentvar.at(iobs),mydata.dec.at(iobs));

	}

	// Here we set up the optimizer and minimize the likelihood
	// LN_COBYLA is a gradient-free optimizer
	// {G,L}{N,D}_xxxx, where G/L denotes global/local optimization and N/D denotes derivative-free/gradient-based algorithms, respectively.
	// http://ab-initio.mit.edu/wiki/index.php/NLopt_Algorithms

	nlopt::opt opt(nlopt::LN_COBYLA, nparam);

	// Give it some guidance about the size of the initial step
	//    vector<double> initialstep(2,0.1);
	//    opt.set_initial_step(initialstep);

	//If there are no bounds on parameters, then this is not necessary. use "HUGE_VAL" if you want it to be unconstrained on one side
	//std::vector<double> lb(nparam,-HUGE_VAL);
	//lb[0] = 0.0;
	//lb[1] = -10.0;
	//lb[2] = 0.0;
	//lb[3] = 0.0;
	//lb[4] = 0.0;
	//lb[5] = 0.0;
	//opt.set_lower_bounds(lb);
	
	//std::vector<double> ub(nparam,HUGE_VAL);
	//ub[0] = 1.0;
	//ub[1] = 0.0;
	//ub[2] = 1.0;
	//ub[3] = 1;
	//ub[4] = 1.5;
	//ub[5] = 1.0;

	//opt.set_upper_bounds(ub);


	// Here we tell the optimizer where it can find the function that will calculate the objective function and the data object
	opt.set_max_objective(objfunc, &mydata);

	//Here we can change settings of the optimizer
	//      opt.set_maxeval(nobs);
	opt.set_xtol_rel(1.0e-10);

	// create variables containing parameters and value of objective
	std::vector<double>param(nparam, 0.0);
	double Value;

	// Set initial values for parameters
	param.at(0) = 0.0; //gamma
	param.at(1) = 0.0; //beta_edu
	param.at(2) = 0.0; //pi+beta_n
	param.at(3) = 1.0; //sigma_epsilon
	param.at(4) = 1.0; //sigma_eta
	param.at(5) = 0.0; //ro

	//Optimize
	opt.optimize(param, Value);

	cout << "Optimum found: gamma=" << param[0] << ", beta_edu=" << param[1] << ", pi+beta_n=" << param[2] << ", sigma_epsilon=" << param[3] << ", sigma_eta=" << param[4] << ", ro=" << param[5]<<" !" << endl;

	return 0;
}


double objfunc(const std::vector<double> &x, std::vector<double> &grad, void*MYdata) {


	//Need to do some fancy stuff ("cast") to get a pointer to the data
	data_estimation *mydata = reinterpret_cast<data_estimation*>(MYdata);

	double objective = 0.0;
	double sigmad0 = sqrt(pow(x[3], 2) + pow(x[4], 2)- 2*x[5]*x[4]*x[3]); //sigma when we solve the CDF and d=0
	double sigmad1 = sqrt((1 - pow(x[5], 2))*pow(x[3], 2)); //sigma when we solve the CDF and d=1


	// here we loop over the observations and add the log-likelihood up
	for (int iobs = 0; iobs < mydata->numobs; iobs++) {

		double mu = (x[3] / x[4])*x[5] * (mydata->wage[iobs] - x[0] * mydata->educ[iobs]); // mean of the CDF when d=1
		
		double CDFd1 = normalCDF(mydata->wage[iobs] - x[0] * mydata->educ[iobs]+ (x[0] - x[1]) * mydata->educ[iobs] - x[2] * mydata->n[iobs],mu ,sigmad1 );
		//double CDFd1 = normalCDF(-(x[0] - x[1]) * mydata->educ[iobs] + x[2] * mydata->n[iobs], mu, sigmad1); //CDF when d=1
		double PDFw = pdf(mydata->wage[iobs] - x[0] * mydata->educ[iobs], x[4]); //Probability of wage
		double CDFd0 = normalCDF(-(x[0] - x[1]) * mydata->educ[iobs] + x[2] * mydata->n[iobs], 0, sigmad0); //CDF when d=0

		//Make sure we don't take the log of zero
		if (CDFd1<1e-10) CDFd1 = 1.0e-10;
		if ((1.0 - CDFd1) < 1e-10) CDFd1 = 1.0 - 1.0e-10;
		if (CDFd0<1e-10) CDFd0 = 1.0e-10;
		if ((1.0 - CDFd0) < 1e-10) CDFd0 = 1.0 - 1.0e-10;
		if (PDFw<1e-10) PDFw = 1.0e-10;
		if ((1.0 - PDFw) < 1e-10) PDFw = 1.0 - 1.0e-10;


		if (mydata->dec[iobs] == 1) {
			objective += (log(CDFd1) + log(PDFw));
		}
		else {
			objective += log(CDFd0);
		}
	}

	printf("Evaluating at %16.10f %16.10f %16.10f,%16.10f,%16.10f,%16.10f, objective=%16.10f\n", x[0], x[1], x[2], x[3], x[4], x[5], objective);

	return objective;
}


double normalCDF(double value,double mu, double sigma)
{
	return  1.0 - 0.5*erfc((value-mu) / (sigma*sqrt(2)));
}

// Returns the probability of x of a gaussian distribution
double pdf(double x, double sigma)
{
	return (1 / (sigma*sqrt(2 * M_PI)))*exp(-0.5*pow(x, 2) / pow(sigma, 2));
}