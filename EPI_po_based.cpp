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


using namespace std;

const int N=500; // number of neurons
const double dt=0.01; //integration step
const double E_na=55.,E_h = +47.,E_ca=+120, E_k=-85, E_cl = -71., V_gaba = -80, V_ampa = 0.;
const double lambda = 10.;
double g_na = 150,g_k=10,g_Ca=0.0,g_A=0.,g_h=0.,g_ahp=0.,tau_ca=400.,Ca_inf = 2.4e-4,a_s=0.25,b_s=0.05;
double g_l_cl, *g_l_k, *meanf, *Vt;
int last_sp[N], T[N], last_poisson_event[N], next_poisson_event[N] , Tp[N];
double G_lg_lg, G_lg_rn, G_rf_rf, G_rf_lg, G_rf_a7;
int N_step1=200000;
int N_step2=2000000; //2000000;
double V[N],m[N],h[N],n[N],m_a[N],h_a[N],m_h[N],m_ca[N],h_ca[N],m_ahp[N],h_ahp[N],Ca[N], se[N], si[N];
double k_v[4][N], k_h[4][N], k_n[4][N], k_m[4][N], k_ma[4][N], k_ha[4][N], k_mh[4][N], k_mca[4][N], k_hca[4][N], k_mahp[4][N], k_hahp[4][N], k_Ca[4][N], k_se[4][N], k_si[4][N];
double I_ext[N],divisor;
double drive0 = 10.0/(2.*96489.);
FILE *dat1, *dat2, *dat3, *dat5;
double k=0.;
double noise_int=0.4;

double S(double x)
{
	return 1./(1+exp(-100*(x-20)));
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
double F(double c)
{
    return c*c*c*c/(200+c*c*c*c); //changed from 0.005
}
double neuromod_to_leak ( double mod_concentr )
{
   // if (mod_concentr<0)
   //     return 0.046;
   // double K =(0.046-0.005)/0.388 ; //coefficient of transfomration
   // double leak = 0.046-K*mod_concentr;
   // if (leak < 0.005) leak = 0.005;
   // if (leak>0.042) leak = 0.042;
   // return leak;
   return 0.0165+(0.024-0.0165)*F(mod_concentr);
}


//ths for voltage
double F_v(int num,double* V, double *k_v,double m,double h, double n, double m_a, double h_a,double m_h, double m_ca, double h_ca,double m_ahp,double h_ahp, double Ca,double I_ext, double* se, double* si, double g_l_k, double g_l_cl)
{
	double I_k = g_k*n*n*n*n*(V[num]+k_v[num]*divisor-E_k);
	double I_na = g_na*m*m*m*h*(V[num]+k_v[num]*divisor-E_na);
	//double I_A = g_A*m_a*h_a*(V[num]+k_v[num]*divisor-E_k);
	//double I_h = g_h*m_h*(V[num]+k_v[num]*divisor-E_h);
	//double I_ahp = g_ahp*m_ahp*m_ahp*h_ahp*(V[num]+k_v[num]*divisor-E_k);
	double I_ahp = g_ahp*m_ahp*(V[num]+k_v[num]*divisor-E_k);
	double I_Ca = g_Ca*m_ca*m_ca*h_ca*(V[num]+k_v[num]*divisor-E_ca);
	double I_leak = g_l_k*(V[num]+k_v[num]*divisor-E_k) + g_l_cl*(V[num]+k_v[num]*divisor-E_cl); //potassium and chloride leak currents
	double I_syn=0.0;
	return -(I_k+I_ahp+I_na+I_leak+I_syn+I_Ca+I_ext);
}
//rhs for gating variable for fast sodium
double F_m(double V,double m)
{
	double Vs = -38.;
	double v = V-Vs;
	double a_m = 0.182*v/ (1. -exp(-v/6.) );
	double b_m = 0.124*(-v)/ ( 1.- exp(v/6.0)  );
	double m_inf = a_m/(a_m+b_m);
	double tau_m = 1./(a_m+b_m)/3.488;
	return (m_inf-m)/tau_m;

}
//rhs for gating variable for fast sodium
double F_h(double V,double h)
{
	double Vh = -66.;
	double v = V-Vh;
	double a_h = -0.015*v/(1-exp(v/6.));
	double b_h = -0.015*(-v)/(1-exp(-(v)/6.)) ;
	double h_inf = a_h/(a_h+b_h);
	double tau_h = 1./(a_h+b_h)/3.488;
	return (h_inf-h)/tau_h;

}
double F_n(double V,double n)
{
//	double Vs = V;
//	double a_n = 0.032*(48.+Vs)/ ( -exp(-(48.+Vs)*0.2) +1. );
//	double b_n = 0.5*exp((-53.-Vs)/40. );
//	double n_inf = a_n/(a_n+b_n);
//	double tau_n=1./(a_n+b_n);
	//if (V<=-60) tau_n = 0.2;
	//return a_n*(1-n)-b_n*n;
	double v = V+30.;
	double a_n = -0.0035*v/(exp(-v/13.)-1);
	double b_n = -0.0035*(-v)/(exp(v/13.)-1) ;

	double n_inf = a_n/(a_n+b_n);
	double tau_n =1.8;
	return (n_inf-n)/tau_n;

}
double F_ma(double V,double m)
{
	double m_inf = 1./(1.+exp(-(V+17.7)/16.6));
	double tau_m = 0.5*1.25;
	return 1./tau_m*(m_inf-m);

}
double F_ha(double V,double h)
{
	double h_inf = 1./(1.+exp((V+72.7)/6.7));
	double tau_h;
	if (V>-63.) tau_h = 6.3;
	else tau_h = 0.3/( exp( 0.2*(V+46.) )+exp( -(V+238.)/37.5 ) );
	return 1./tau_h*(h_inf-h);

}
double F_mh(double V,double m)
{
	double m_inf = 1./(1.+exp((V+110.)/7.8));
	double tau_m =5.3+20./(exp((V+75.)/11.)+exp(-(V+110.)/37.));
	return 1./tau_m*(m_inf-m);

}
double F_mca(double V, double m)
{
	double a = 1.0*0.055*(-27 - V)/(exp((-27-V)/3.8) - 1);
	double b = 1.0*0.94*exp((-75-V)/17);
	double tau_m = (1/(a+b))/2.953;
	double m_inf = a/(a+b);
	return (m_inf - m)/tau_m;

}
double F_hca(double V, double h)
{
	double a = 1.*0.000457*exp((-13-V)/50);
	double b = 1.0*0.0065/(exp((-V-15)/28) + 1);
	double tau_h = (1/(a+b))/2.953;
	double h_inf = a/(a+b);

	return (h_inf - h)/tau_h;
}


double F_Ca(double V,double Ca, double I_Ca)
{
	double drive = -drive0*I_Ca;
	if (drive<0) drive = 0.;
	return drive+(Ca_inf-Ca)/tau_ca;
}

double F_mahp(double V, double m, double Ca)
{
	double a = 0.01 * Ca;
	double b = 0.02;
	double tau_m = (1/(a+b))/2.953;
	double m_inf = 2.953*a/(a+b);
	return -(1/tau_m)*(m - m_inf);
//	double tau_m;
//	if (V<-60.)
// 		tau_m = (1.25+175.03*exp(0.026*(V+10.)))/3.488;
//	else
// 		tau_m = (1.25+13.0*exp(-0.026*(V+10.)))/3.488;
//	double m_inf = 1./(1+exp(-(V+11)/12.));
//	return -(1./tau_m)*(m - m_inf);
}
double F_hahp(double V, double h, double Ca)
{
	double tau_h;
 	tau_h = (360.+(1010+24.*(V+65.))*exp(-((V+85.)/48.)*((V+85.)/48.))  )/3.488;
	double h_inf = 1./(1+exp((V+64)/11.));
	return -(1./tau_h)*(h - h_inf);
}
double F_si(int T, double s)
{
        return a_s*T*(1-s)-b_s*s;

}
double F_se(int T,double s)
{
        return 1.2*a_s*T*(1-s)-2.*b_s*s;
}

void solve(int Tau, FILE* dat1, FILE* dat2, FILE* dat5, int output, double* g_l_k, double g_l_cl)
{

	for (int counter=0;counter<Tau;counter++) //loop over time
	{
		if (counter%1000==0)
			cerr << (double)counter/(double)Tau << endl;

		divisor=0.;
		#pragma omp parallel for
		for (int i=0;i<N;i++)
		{
			if (counter-last_sp[i]<100) T[i]=1;
			else T[i] = 0;

			double I_ca = g_Ca*m_ca[i]*m_ca[i]*h_ca[i]*(V[i]-E_ca);
			k_v[0][i] = F_v(i,V,k_v[0],m[i],h[i],n[i],m_a[i],h_a[i],m_h[i],m_ca[i],h_ca[i],m_ahp[i],h_ahp[i],Ca[i],I_ext[i], se, si, g_l_k[counter], g_l_cl)*dt;
			k_m[0][i] = F_m(V[i],m[i])*dt;
			k_h[0][i] = F_h(V[i],h[i])*dt;
			k_n[0][i] = F_n(V[i],n[i])*dt;
		//	k_ma[0][i] = F_ma(V[i],m_a[i])*dt;
		//	k_ha[0][i] = F_ha(V[i],h_a[i])*dt;
		//	k_mh[0][i] = F_mh(V[i],m_h[i])*dt;
			k_mca[0][i] = F_mca(V[i],m_ca[i])*dt;
			k_hca[0][i] = F_hca(V[i],h_ca[i])*dt;
			k_mahp[0][i] = F_mahp(V[i],m_ahp[i],Ca[i])*dt;
			//k_hahp[0][i] = F_hahp(V[i],h_ahp[i],Ca[i])*dt;
			k_Ca[0][i] = F_Ca(V[i],Ca[i],I_ca)*dt;
			k_se[0][i] = F_se(T[i],se[i])*dt;
			k_si[0][i] = F_si(T[i],si[i])*dt;
		}

		divisor=0.5;
		#pragma omp parallel for
		for (int i=0;i<N;i++)
		{
		    double I_ca = g_Ca*m_ca[i]*m_ca[i]*h_ca[i]*(V[i]-E_ca);
			k_v[1][i] = F_v(i,V,k_v[0],m[i]+k_m[0][i]/2.0,h[i]+k_h[0][i]/2.0,n[i]+k_n[0][i]/2.0,m_a[i]+k_ma[0][i]/2.0,h_a[i]+k_ha[0][i]/2.0,m_h[i]+k_mh[0][i]/2.0,m_ca[i]+k_mca[0][i]/2.0,h_ca[i]+k_hca[0][i]/2.0,m_ahp[i]+k_mahp[0][i]/2.0,h_ahp[i]+k_hahp[0][i]/2.0,Ca[i]+k_Ca[0][i]/2.0,I_ext[i],se,si, g_l_k[counter], g_l_cl)*dt;
			k_m[1][i] = F_m(V[i]+k_v[0][i]/2.0,m[i]+k_m[0][i]/2.0)*dt;
			k_h[1][i] = F_h(V[i]+k_v[0][i]/2.0,h[i]+k_h[0][i]/2.0)*dt;
			k_n[1][i] = F_n(V[i]+k_v[0][i]/2.0,n[i]+k_n[0][i]/2.0)*dt;
		//	k_ma[1][i] = F_ma(V[i]+k_v[0][i]/2.0,m_a[i]+k_ma[0][i]/2.0)*dt;
		//	k_ha[1][i] = F_ha(V[i]+k_v[0][i]/2.0,h_a[i]+k_ha[0][i]/2.0)*dt;
		//	k_mh[1][i] = F_mh(V[i]+k_v[0][i]/2.0,m_h[i]+k_mh[0][i]/2.0)*dt;
			k_mca[1][i] = F_mca(V[i]+k_v[0][i]/2.0,m_ca[i]+k_mca[0][i]/2.0)*dt;
			k_hca[1][i] = F_hca(V[i]+k_v[0][i]/2.0,h_ca[i]+k_hca[0][i]/2.0)*dt;
			k_mahp[1][i] = F_mahp(V[i]+k_v[0][i]/2.0,m_ahp[i]+k_mahp[0][i]/2.0,Ca[i]+k_Ca[0][i]/2.)*dt;
			//k_hahp[1][i] = F_hahp(V[i]+k_v[0][i]/2.0,h_ahp[i]+k_hahp[0][i]/2.0,Ca[i]+k_Ca[0][i]/2.)*dt;
			k_Ca[1][i] = F_Ca(V[i]+k_v[0][i]/2.0,Ca[i]+k_Ca[0][i]/2.,I_ca)*dt;
			k_se[1][i] = F_se(T[i],se[i]+k_se[0][i]/2.)*dt;
			k_si[1][i] = F_si(T[i],si[i]+k_si[0][i]/2.)*dt;
		}


		#pragma omp parallel for
		for (int i=0;i<N;i++)
		{
		    double I_ca = g_Ca*m_ca[i]*m_ca[i]*h_ca[i]*(V[i]-E_ca);
			k_v[2][i] = F_v(i,V,k_v[1],m[i]+k_m[1][i]/2.0,h[i]+k_h[1][i]/2.0,n[i]+k_n[1][i]/2.0,m_a[i]+k_ma[1][i]/2.0,h_a[i]+k_ha[1][i]/2.0,m_h[i]+k_mh[1][i]/2.0,m_ca[i]+k_mca[1][i]/2.0,h_ca[i]+k_hca[1][i]/2.0,m_ahp[i]+k_mahp[1][i]/2.,h_ahp[i]+k_hahp[1][i]/2.,Ca[i]+k_Ca[1][i]/2.0,I_ext[i], se, si, g_l_k[counter], g_l_cl)*dt;
			k_m[2][i] = F_m(V[i]+k_v[1][i]/2.0,m[i]+k_m[1][i]/2.0)*dt;
			k_h[2][i] = F_h(V[i]+k_v[1][i]/2.0,h[i]+k_h[1][i]/2.0)*dt;
			k_n[2][i] = F_n(V[i]+k_v[1][i]/2.0,n[i]+k_n[1][i]/2.0)*dt;
		//	k_ma[2][i] = F_ma(V[i]+k_v[1][i]/2.0,m_a[i]+k_ma[1][i]/2.0)*dt;
		//	k_ha[2][i] = F_ha(V[i]+k_v[1][i]/2.0,h_a[i]+k_ha[1][i]/2.0)*dt;
		//	k_mh[2][i] = F_mh(V[i]+k_v[1][i]/2.0,m_h[i]+k_mh[1][i]/2.0)*dt;
			k_mca[2][i] = F_mca(V[i]+k_v[1][i]/2.0,m_ca[i]+k_mca[1][i]/2.0)*dt;
			k_hca[2][i] = F_hca(V[i]+k_v[1][i]/2.0,h_ca[i]+k_hca[1][i]/2.0)*dt;
			k_mahp[2][i] = F_mahp(V[i]+k_v[1][i]/2.0,m_ahp[i]+k_mahp[1][i]/2.0,Ca[i]+k_Ca[1][i]/2.)*dt;
			//k_hahp[2][i] = F_mahp(V[i]+k_v[1][i]/2.0,h_ahp[i]+k_hahp[1][i]/2.0,Ca[i]+k_Ca[1][i]/2.)*dt;
			k_Ca[2][i] = F_Ca(V[i]+k_v[1][i]/2.0,Ca[i]+k_Ca[1][i]/2.,I_ca)*dt;
			k_se[2][i] = F_se(T[i],se[i]+k_se[1][i]/2.)*dt;
			k_si[2][i] = F_si(T[i],si[i]+k_si[1][i]/2.)*dt;
		}

		divisor=1.;
		#pragma omp parallel for
		for (int i=0;i<N;i++)
		{
		    double I_ca = g_Ca*m_ca[i]*m_ca[i]*h_ca[i]*(V[i]-E_ca);
			k_v[3][i] = F_v(i,V,k_v[2],m[i]+k_m[2][i],h[i]+k_h[2][i],n[i]+k_n[2][i],m_a[i]+k_ma[2][i],h_a[i]+k_ha[2][i],m_h[i]+k_mh[2][i],m_ca[i]+k_mca[2][i],h_ca[i]+k_hca[2][i],m_ahp[i]+k_mahp[2][i],h_ahp[i]+k_hahp[2][i],Ca[i]+k_Ca[2][i],I_ext[i], se, si, g_l_k[counter], g_l_cl)*dt;
			k_m[3][i] = F_m(V[i]+k_v[2][i],m[i]+k_m[2][i])*dt;
			k_h[3][i] = F_h(V[i]+k_v[2][i],h[i]+k_h[2][i])*dt;
			k_n[3][i] = F_n(V[i]+k_v[2][i],n[i]+k_n[2][i])*dt;
		//	k_ma[3][i] = F_ma(V[i]+k_v[2][i],m_a[i]+k_ma[2][i])*dt;
		//	k_ha[3][i] = F_ha(V[i]+k_v[2][i],h_a[i]+k_ha[2][i])*dt;
		//	k_mh[3][i] = F_mh(V[i]+k_v[2][i],m_h[i]+k_mh[2][i])*dt;
			k_mca[3][i] = F_mca(V[i]+k_v[2][i],m_ca[i]+k_mca[2][i])*dt;
			k_hca[3][i] = F_hca(V[i]+k_v[2][i],h_ca[i]+k_hca[2][i])*dt;
			k_mahp[3][i] = F_mahp(V[i]+k_v[2][i],m_ahp[i]+k_mahp[2][i],Ca[i]+k_Ca[2][i]/2.)*dt;
			//k_hahp[3][i] = F_hahp(V[i]+k_v[2][i],h_ahp[i]+k_hahp[2][i],Ca[i]+k_Ca[2][i]/2.)*dt;
			k_Ca[3][i] = F_Ca(V[i]+k_v[2][i],Ca[i]+k_Ca[2][i]/2.,I_ca)*dt;
			k_se[3][i] = F_se(T[i],se[i]+k_se[2][i])*dt;
			k_si[3][i] = F_si(T[i],si[i]+k_si[2][i])*dt;
		}

        double mf = 0.;
		#pragma omp parallel for reduction( + : mf )
		for (int i=0;i<N;i++)
		{
			V[i] += (k_v[0][i]+2.0*k_v[1][i]+2.0*k_v[2][i]+k_v[3][i])/6.0;
			m[i] += (k_m[0][i]+2.0*k_m[1][i]+2.0*k_m[2][i]+k_m[3][i])/6.0;
			h[i] += (k_h[0][i]+2.0*k_h[1][i]+2.0*k_h[2][i]+k_h[3][i])/6.0;
			n[i] += (k_n[0][i]+2.0*k_n[1][i]+2.0*k_n[2][i]+k_n[3][i])/6.0;
		//	m_a[i] += (k_ma[0][i]+2.0*k_ma[1][i]+2.0*k_ma[2][i]+k_ma[3][i])/6.0;
		//	h_a[i] += (k_ha[0][i]+2.0*k_ha[1][i]+2.0*k_ha[2][i]+k_ha[3][i])/6.0;
		//	m_h[i] += (k_mh[0][i]+2.0*k_mh[1][i]+2.0*k_mh[2][i]+k_mh[3][i])/6.0;
			m_ca[i] += (k_mca[0][i]+2.0*k_mca[1][i]+2.0*k_mca[2][i]+k_mca[3][i])/6.0;
			h_ca[i] += (k_hca[0][i]+2.0*k_hca[1][i]+2.0*k_hca[2][i]+k_hca[3][i])/6.0;
			m_ahp[i] += (k_mahp[0][i]+2.0*k_mahp[1][i]+2.0*k_mahp[2][i]+k_mahp[3][i])/6.0;
			//h_ahp[i] += (k_hahp[0][i]+2.0*k_hahp[1][i]+2.0*k_hahp[2][i]+k_hahp[3][i])/6.0;
			Ca[i] += (k_Ca[0][i]+2.0*k_Ca[1][i]+2.0*k_Ca[2][i]+k_Ca[3][i])/6.0;
			se[i] += (k_se[0][i]+2.0*k_se[1][i]+2.0*k_se[2][i]+k_se[3][i])/6.0;
			si[i] += (k_si[0][i]+2.0*k_si[1][i]+2.0*k_si[2][i]+k_si[3][i])/6.0;

			V[i] = V[i] + noise_int*sqrt(dt)*gauss_gen();

			mf+= 1./((double)N)*si[i];

                        if (i==0) Vt[counter]=V[i];

			if ((V[i]>-20.) && (counter - last_sp[i]>400))
			{
				last_sp[i] = counter;
                if ((output)) //removed &&(i==0)
                    fprintf(dat1, "%i %i\n",i, counter);
			}
		}
        if (output)
		{
            meanf[counter] = mf;
		}
	}

}
void charstr_to_int(const char* inp, int& outp)
{
	stringstream ss;
	ss << inp;
	ss >> outp;
}
int main(int argc, char* argv[])
{
	
    //argv[1] - seed number
    int seed;
    charstr_to_int(argv[1], seed);
    srand(seed);

	string fname1 = string("./data/seed")+string(argv[1])+string("/data_spks_")+string(argv[2]);
	string fname2 = string("./data/seed")+string(argv[1])+string("/data_meanf_")+string(argv[2]);
	string fname3 = string("../../A7_and_RN_neurons/data/seed")+string(argv[1])+string("/sIN_")+string(argv[2]);
        string fname5 = string("./data/seed")+string(argv[1])+string("/V")+string("_")+string(argv[2]);
	
	omp_set_num_threads(8);
	
//	g_l_cl= 0.001+0.02*ind_Cl_leak/((double)grid_Cl);
//	g_l_k = 0.001+0.02*ind_K_leak/((double)grid_K);
	g_l_cl= 0.015;
    g_l_k = new double[N_step2];
    meanf = new double[N_step2];
    Vt = new double[N_step2];

    dat3 = fopen(fname3.c_str(),"rb");
    int readed = fread ( g_l_k, sizeof(double), N_step2, dat3); //actually g_l_k is sIN here
    if (readed != N_step2)
        cerr << "Warning, not all elements were readed" << endl;
    for (int i=0;i<N_step2;i++)
    {
			g_l_k[i] = neuromod_to_leak(g_l_k[i]); // I removed ./5000
			if (i%10000==0)
				cerr << "leak: " << g_l_k[i] << " " << i << endl;
            
    }
	
	for (int i=0;i<N;i++)
	{
		m[i]=n[i]=h[i]=Ca[i]=m_a[i]=h_a[i]=m_h[i]=m_ca[i]=h_ca[i] =m_ahp[i] = h_ahp[i]= 0.01;
		se[i] = si[i] = 0.;
		last_sp[i] = -10000;
		last_poisson_event[i] = -10000;
		Ca[i]=0.0001;
		V[i] = -67.0;
		I_ext[i]=-0.0;
	}
	dat1 = fopen(fname1.c_str(), "w");
	dat2 = fopen(fname2.c_str(), "wb");
        dat5 = fopen(fname5.c_str(), "wb");

	solve(N_step1/10, dat1, dat2, dat5, 0, g_l_k, g_l_cl);
	for (int i=0;i<N;i++)
		I_ext[i] = -0.6;
	solve(N_step2, dat1, dat2, dat5, 1, g_l_k, g_l_cl);
	fwrite( meanf, sizeof(double), N_step2, dat2 );
	fwrite( Vt, sizeof(double), N_step2, dat5 );

	fclose(dat1);
	fclose(dat2);
        fclose(dat5);
    delete[] g_l_k;
	

	return 0;
}
