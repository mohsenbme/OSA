#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <omp.h>
#include <random>
#include <ctime>
#include <list>


using namespace std;

const int N=5000;
const double dt=0.01;
int T[N], r[N];
//const double a = 1/100., b = 1./1000., K3 = 1.8/1000., K4 = 0.34/1000.;
const double a = 1/100., b = 1./1000., K3 = 0.18, K4 = 0.034;
int N_step=4000000; //4000000
double *PO = new double[N_step];
double *PC = new double[N_step];
double *PB = new double[N_step];
double *POin = new double[N_step];
double *meanf = new double[N_step];
double  *sNE = new double[N_step];
double  *sIN = new double[N_step];
FILE *dat1;
FILE *datpo;

//void make_fname(const string& prefix, const char* inp,  string& outp)
//{
//    stringstream ss;
//    ss << prefix << inp;
//    ss >> outp;
//}
void get_pair(const string& line, pair<int,int>& p)
{
    stringstream ss;
	ss << line;
	ss >> p.first;
	ss >> p.second;
	
}
double gauss_gen()
{
        double rsq,x1,x2;
        do {
                x1 = -1.+2.*((double)rand())/((double)RAND_MAX);
                x2 = -1.+2.*((double)rand())/((double)RAND_MAX);
                rsq = x1*x1+x2*x2;
        }while((rsq>=1.0) || (rsq == 0.0));
        return x1*sqrt(-2.*log(rsq)/rsq);

}


double F_s(double s, double p)
{
    return K3*p-K4*s;
}
double F_r( double r)
{
        
        return -b*r;
}


void gen_neuromod(int Tau, const list<int> *spikes) 
{
	//double** s = new double*[N];
	//double s[N];
	double r[N];

//	for (int i=0;i<N;i++)
//		s[i] = new double[Tau+1]();
	for (int i=0;i<N;i++)
	{
     //   s[i] = 0.;
		r[i] = 0.;
	}
	for (int counter=0;counter<Tau;counter++)
        meanf[counter]=0.;
       
    //#pragma omp parallel for
	for (int i=0; i<N; i++)
	{
        if (i%100==0)
            cerr << i << endl;
	//    for (int counter=0;counter<Tau;counter++)
	//		s[i][counter] = 0;

		list<int>::const_iterator it = spikes[i].begin();
        if (i==0) cerr << "size: " << spikes[0].size() << endl;



	    for (int counter=0;counter<Tau;counter++)
		{
			if (counter == *it)
			{
                if (i==0) cerr << counter << endl;
				T[i] = 300;
				if (it != spikes[i].end()) ++it;
			}
			if (T[i]>0)
			{
                r[i]=1.;
				T[i]--;
                                meanf[counter] +=1;

			}
			else
            {
				r[i] = r[i] + F_r(r[i])*dt;
            }
			//s[i] = s[i] + F_s(s[i], r[i])*dt;
            //meanf[counter] += s[i];
        	//out_file << mean/(double)N << endl;
		}
        

	}
        double alpha=0; // function of NE/5-HT
        double beta=5, delta=5;
        double gamma=6;  // function of Pz init:6
for (int counter=0;counter<N_step;counter++)
	{
		alpha += meanf[counter]/N_step;
	}
cerr << "alpha: " << alpha << endl;
PO[0]=0.8; POin[0]=0.8;
PB[0]=0.05;
for (int counter=0;counter<N_step-1;counter++)
	{
          PO[counter+1] = PO[counter] + (alpha*(1 - PB[counter]) - (alpha + beta)*PO[counter])*dt;
         // PC[counter+1] = PC[counter] + (beta*PO[counter] + delta*PB[counter] - gamma*PC[counter] - alpha*PC[counter])*dt;
          PB[counter+1] = PB[counter] + (gamma*(1 - PO[counter]) - (gamma + delta)*PB[counter])*dt;
        }
double PONEss=PO[N_step-1];
double PBNEss=PB[N_step-1];
double PCNEss=1-(PONEss+PBNEss);
cerr << "POss: " << PONEss<< endl;
for (int counter=0;counter<N_step-1;counter++)
sNE[counter+1]=sNE[counter] + (0.18*PONEss-0.034*sNE[counter])*dt;

cerr << "sNE: " << sNE[N_step-1]<< endl;

double alphaME=0; // function of ME
double betaIN=5;
//for (int counter=0;counter<N_step-1;counter++)
         //POin[counter+1] = POin[counter] + ( (alpha + alphaME)*(1 - POin[counter]) - betaIN*POin[counter])*dt;
         // POin[counter]=(alpha + alphaME)/(alpha + alphaME+betaIN); //ss
double POINss=(alpha + alphaME)/(alpha + alphaME+betaIN); //POin[N_step-1];
cerr << "POINss: " << POINss<< endl;
for (int counter=0;counter<N_step-1;counter++)
sIN[counter+1]=sIN[counter] + (0.18*POINss-0.034*sIN[counter])*dt;


}

	
	
int main(int argc, char* argv[])
{
    //argv[1] - seed number
    //argv[2] - characteristic interval
    srand((int)time(NULL));
	omp_set_num_threads(32);
    list <int> *spikes;
	spikes = new list <int>[N];

    string path = "/home/naji/OSA/code/A7_and_RN_neurons";
    string fname_in  = path+string("/data/seed")+string(argv[1])+string("/data_")+string(argv[2]);
    string fname_out = path+string("/data/seed")+string(argv[1])+string("/sNE_")+string(argv[2]);
    string fname_out2 = path+string("/data/seed")+string(argv[1])+string("/sIN_")+string(argv[2]);
    //string fname_outpo = path+string("/data/seed")+string(argv[1])+string("/mnfldPO_")+string(argv[2]);

    ifstream dat1 (fname_in.c_str());
	if (dat1.is_open())
	{
		string line;
		while ( getline (dat1,line) )
		{
			pair<int,int> p; //p.first is index, p.second is time
			get_pair(line, p);
			spikes[p.first].push_back(p.second);
		}
	}
    cerr << "size: " << spikes[0].size() << endl;
	dat1.close();

	gen_neuromod(N_step, spikes);
    FILE *dat2 = fopen(fname_out.c_str(),"wb");
    fwrite(&sNE[N_step/2], sizeof(double), N_step/2, dat2);
    FILE *dat3 = fopen(fname_out2.c_str(),"wb");
    fwrite(&sIN[N_step/2], sizeof(double), N_step/2, dat3);
    
    //FILE *datpo = fopen(fname_outpo.c_str(),"wb");
    //fwrite(&PO[N_step], sizeof(double), N_step, datpo);
	fclose(dat2);
        fclose(dat3);
//fclose(datpo);
    
	delete[] spikes;
	delete[] meanf;

	return 0;
}
