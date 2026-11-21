/*
    This is a header file contains fundamental constants that should not need to
    be changed by the user.

    Do a text search to find parameters from a specific .H file from 21cmFAST
    (i.e. INIT_PARAMS.H, COSMOLOGY.H, ANAL_PARAMS.H and HEAT_PARAMS)

    NOTE: Not all 21cmFAST variables will be found below. Only those useful for 21CMMC

 */

// ----------------------------------------------------------------------------------------- //

// Taken from ANAL_PARAMS.H

// ----------------------------------------------------------------------------------------- //

#define L_FACTOR (float) (0.620350491) // factor relating cube length to filter radius = (4PI/3)^(-1/3)

// ----------------------------------------------------------------------------------------- //

// Taken from elec_interp.c

// ----------------------------------------------------------------------------------------- //



/*
 Filenames of the appropriate output from RECFAST to be used as boundary conditions in Ts.c
 as well as other tables used to compute the spin temperature
 */
#define RECFAST_FILENAME (const char *) "recfast_LCDM.dat"
#define STELLAR_SPECTRA_FILENAME (const char *) "stellar_spectra.dat"
#define KAPPA_EH_FILENAME (const char *) "kappa_eH_table.dat"
#define KAPPA_PH_FILENAME (const char *) "kappa_pH_table.dat"
//Interpolation table for Lya heating efficiencies
#define LYA_HEATING_FILENAME (const char *) "Lyman_alpha_heating_table.dat"


// ----------------------------------------------------------------------------------------- //

// Taken from HEAT_PARAMS.H

// ----------------------------------------------------------------------------------------- //


/* Maximum allowed value for the kinetic temperature. Useful to set to avoid some spurious behaviour
 when the code is run with redshift poor resolution and very high X-ray heating efficiency */
#define MAX_TK (float) 5e4


// ----------------------------------------------------------------------------------------- //

// Taken from COSMOLOGY.H

// ----------------------------------------------------------------------------------------- //

// STRUCTURE //
#define Deltac (1.68) // at z=0, density excess at virialization
#define N_nu (1.0) // # of heavy neutrinos (for EH trasfer function)
#define BODE_e (0.361) // Epsilon parameter in Bode et al. 2000 trans. funct.
#define BODE_n (5.0) // Eda parameter in Bode et al. 2000 trans. funct.
#define BODE_v (1.2) // Nu parameter in Bode et al. 2000 trans. funct.
// SHETH params gotten from barkana et al. 2001, pg. 487
#define SHETH_a (0.73) // Sheth and Tormen a parameter (from Jenkins et al. 2001)
#define SHETH_p (0.175) // Sheth and Tormen p parameter (from Jenkins et al. 2001)
#define SHETH_A (0.353) // Sheth and Tormen A parameter (from Jenkins et al. 2001)


#define SIGMAVCB (29.0) //rms value of the DM-b relative velocity [im km/s]


// Universal FOF HMF (Watson et al. 2013)
#define Watson_A (0.282) // Watson FOF HMF, A parameter (Watson et al. 2013)
#define Watson_alpha (2.163) // Watson FOF HMF, alpha parameter (Watson et al. 2013)
#define Watson_beta (1.406) // Watson FOF HMF, beta parameter (Watson et al. 2013)
#define Watson_gamma (1.210) // Watson FOF HMF, gamma parameter (Watson et al. 2013)

// Universal FOF HMF with redshift evolution (Watson et al. 2013)
#define Watson_A_z_1 (0.990) // Watson FOF HMF, normalisation of A_z parameter (Watson et al. 2013)
#define Watson_A_z_2 (-3.216) // Watson FOF HMF, power law of A_z parameter (Watson et al. 2013)
#define Watson_A_z_3 (0.074) // Watson FOF HMF, offset of A_z parameter (Watson et al. 2013)
#define Watson_alpha_z_1 (5.907) // Watson FOF HMF, normalisation of alpha_z parameter (Watson et al. 2013)
#define Watson_alpha_z_2 (-3.058) // Watson FOF HMF, power law of alpha_z parameter (Watson et al. 2013)
#define Watson_alpha_z_3 (2.349) // Watson FOF HMF, offset of beta_z parameter (Watson et al. 2013)
#define Watson_beta_z_1 (3.136) // Watson FOF HMF, normalisation of beta_z parameter (Watson et al. 2013)
#define Watson_beta_z_2 (-3.599) // Watson FOF HMF, power law of beta_z parameter (Watson et al. 2013)
#define Watson_beta_z_3 (2.344) // Watson FOF HMF, offset of beta_z parameter (Watson et al. 2013)
#define Watson_gamma_z (1.318) // Watson FOF HMF, gamma parameter (Watson et al. 2013)


// CONSTANTS //
#define LN10 (double) (2.30258509299)
#define SIGMAT (double) (6.6524e-25)  // Thomson scattering cross section in cm^-2
#define SIGMA_HI (double) (6.3e-18)  // HI ionization  cross section at 13.6 eV in cm^-2
#define E (double) (2.71828182846)
#define PI (double) (3.14159265358979323846264338327)
#define TWOPI (double) (2.0*PI)
#define FOURPI (double) (4.0*PI)
#define G (double) 6.67259e-8 // cm^3 g^-1 s^-2
#define hplank (double) 6.62606896e-27 // erg s
#define TINY (double) 1e-30
#define FRACT_FLOAT_ERR (double) 1e-7 // fractional floating point error
#define f_alpha (float) 0.4162 // oscillator strength of Lya
#define Ly_alpha_HZ  (double ) 2.46606727e15  // frequency of Lyalpha
#define C  (double) 29979245800.0  //  speed of light  (cm/s)
#define C_KMS (double) C/1e5  /* speed of light in km/s  */
#define alphaA_10k (double) 4.18e-13 // taken from osterbrock for T=10000
#define alphaB_10k (double) 2.59e-13 // taken from osterbrock for T=10000
#define alphaB_20k (double) 2.52e-13 // taken from osterbrock for T=20000
#define Ly_alpha_ANG (double) 1215.67
#define Ly_beta_ANG (double) 1025.18
#define Ly_gamma_ANG (double) 972.02
#define NV_ANG (double) 1240.81 // NV line center
#define CMperMPC (double) 3.086e24 // cm/Mpc
#define SperYR (double) 31556925.9747 // s/yr
#define Msun (double) 1.989e33 // g
#define Rsun (double) 6.9598e10 // cm
#define Lsun (double) 3.90e33 // erg/s
#define T_cmb (double) 2.728 // K
#define k_B (double) 1.380658e-16 // erg / K
#define m_p (double) 1.6726231e-24 // proton mass (g)
#define m_e (double) 9.10938188e-28 // electron mass (g)
#define e_charge (double) 4.80320467e-10 // elemetary charge (esu=g^1/2 cm^3/2 s^-1
#define SQDEG_ALLSKY (double) ((360.0*360.0)/PI) // Square degrees in all sky
#define G_AB_Jy (double) 3631.0 // AB mag constant in Jy
#define NU_over_EV (double) (1.60217646e-12 / hplank)
#define NU_LW_THRESH (double) (11.18*NU_over_EV)
#define NUIONIZATION (double) (13.60*NU_over_EV)  // ionization frequency of H
#define HeII_NUIONIZATION (double) (NUIONIZATION*4) // ionization frequency of HeII
#define HeI_NUIONIZATION (double) (24.59*NU_over_EV) // ionization frequency of HeI
#define T21 (double) 0.0682 // temperature corresponding to the 21cm photon
#define A10_HYPERFINE (double) 2.85e-15 // spontaneous emission coefficient in s^-1

#define Lambda_21 (double) 21.106114054160 //Wavelength of 21cm Radiation in cm
#define A21_Lya (double) 6.24e8 //Spontaneous emission coefficient for Lyman-Alpha line in s^-1

#define vac_perm (double) 8.8541878128e-12 // vacuum permittivity in farads m^-1


// ----------------------------------------------------------------------------------------- //

// Taken from heating_helper_progs.c

// ----------------------------------------------------------------------------------------- //



#define NSPEC_MAX (int) 23
#define RECFAST_NPTS (int) 501
#define KAPPA_10_NPTS (int) 27
#define KAPPA_10_elec_NPTS (int) 20
#define KAPPA_10_pH_NPTS (int) 17

#define KAPPA_10_NPTS_Spline (int) 30
#define KAPPA_10_elec_NPTS_Spline (int) 30
#define KAPPA_10_pH_NPTS_Spline (int) 30

#define zpp_interp_points_SFR (int) (400)                  /* Number of interpolation points for the interpolation table for z'' */
#define dens_Ninterp (int) (400)                       /* Number of interpolation points for the interpolation table for the value of the density field */

//Variables needed to read the Lyman-Alpha heating table
#define Tk_max (double) 3.0 //log10 (Tk_max)
#define Tk_min (double) -1.0 //log10 (Tk_min)
#define Ts_max (double) 3.0 //log10 (Ts_max)
#define Ts_min (double) -1.0 //log10 (Ts_min)
#define nT (int) 101
#define taugp_max (double) 7.0 //log10 (taugp_max)
#define taugp_min (double) 1.0 //log10 (taugp_min)
#define ngp (int) 51

// ----------------------------------------------------------------------------------------- //

// Taken from elec_interp.c

// ----------------------------------------------------------------------------------------- //



#define x_int_NXHII  14
#define x_int_NENERGY  258



// ----------------------------------------------------------------------------------------- //

// From ps.c (temporary way to including transfer function from CLASS

// ----------------------------------------------------------------------------------------- //


#define CLASS_FILENAME (const char *) "Transfers_z0.dat"
#define CLASS_LENGTH  150 //length of the CLASS transfer function
#define KBOT_CLASS (float) (1e-5) //max and min k in  CLASS transfer function, temporary until interfaced properly
#define KTOP_CLASS (float) (1e3)
//parameters for the M(sigma) power-law relation for FAST_FCOLL_TABLES
#define MPIVOT1 (double) (1.5e9) //pivot masses
#define MPIVOT2 (double) (5.3e5)
#define AINDEX1 (double) (9.0) //power-law indices for low, intermediate, and high masses
#define AINDEX2 (double) (13.6)
#define AINDEX3 (double) (21.0)
#define MMIN_FAST (double) (1e5) // min mass at which the sigma table is computed if FAST_FCOLL_TABLES is turned on. Has to be below MPIVOT2
//parameters for DM-baryon relative velocity effect on the power spectrum
#define KP_VCB_PM (300.0) //Mpc-1
#define A_VCB_PM (0.24)
#define SIGMAK_VCB_PM (0.9)
//this is for vcb=vrms at z=20. It scales roughly as sqrt(v) and (1+z)^(-1/6.)


// ----------------------------------------------------------------------------------------- //

// For reionization_feedback, reference Sobacchi & Mesinger 2013

// ----------------------------------------------------------------------------------------- //

#define HALO_BIAS     (double) 2.0
#define REION_SM13_M0 (double) 3e9
#define REION_SM13_A  (double) 0.17
#define REION_SM13_B  (double) -2.1
#define REION_SM13_C  (double) 2.0
#define REION_SM13_D  (double) 2.5

__device__ double kappa_10(double TK, int flag)
{

    double ans;
    int tkin_spline_int;

    const double BinWidth_10=0.317597943861;
    const double inv_BinWidth_10=1./0.317597943861;
    const double tkin_spline[30]={0.0,0.317597943861,0.635195887722,0.952793831584,1.27039177544,1.58798971931,1.90558766317,2.22318560703,2.54078355089,2.85838149475,3.17597943861,3.49357738247,3.81117532633,4.1287732702,4.44637121406,4.76396915792,5.08156710178,5.39916504564,5.7167629895,6.03436093336,6.35195887722,6.66955682109,6.98715476495,7.30475270881,7.62235065267,7.93994859653,8.25754654039,8.57514448425,8.89274242811,9.21034037198};
    const double kap_spline[30]={-29.6115227098,-29.6228184691,-29.5917673123,-29.4469989515,-29.1171430989,-28.5382192456,-27.7424388865,-26.8137036254,-25.8749225449,-25.0548322235,-24.4005076336,-23.8952335377,-23.5075651004,-23.201637629,-22.9593758343,-22.7534867461,-22.5745752086,-22.4195690855,-22.2833176123,-22.1549519419,-22.0323282988,-21.9149994721,-21.800121439,-21.6839502137,-21.5662434981,-21.4473595491,-21.3279560712,-21.2067614838,-21.0835560288,-20.9627928675};


    TK = log(TK);

    if (TK < tkin_spline[0]) { // Below 1 K, just use that value
        ans = kap_spline[0];
    } else if (TK > tkin_spline[KAPPA_10_NPTS_Spline-1]) {
        // Power law extrapolation
        ans = log(exp(kap_spline[KAPPA_10_NPTS_Spline-1])*pow(exp(TK)/exp(tkin_spline[KAPPA_10_NPTS_Spline-1]),0.381));
    } else { // Do spline

        tkin_spline_int = (int)floor((TK - tkin_spline[0])*inv_BinWidth_10);

        ans = kap_spline[tkin_spline_int] + ( TK - (tkin_spline[0] + BinWidth_10*(float)tkin_spline_int) )*( kap_spline[tkin_spline_int+1] - kap_spline[tkin_spline_int] )*inv_BinWidth_10;
    }

    return exp(ans);
}
__device__ double kappa_10_pH(double T, int flag)
{
    double ans;
    int TK_spline_int;

    const double BinWidth_pH=0.341499570777;
    const double inv_BinWidth_pH=1./0.341499570777;
    const double TK_spline[30]={0.0,0.341499570777,0.682999141554,1.02449871233,1.36599828311,1.70749785389,2.04899742466,2.39049699544,2.73199656622,3.07349613699,3.41499570777,3.75649527855,4.09799484933,4.4394944201,4.78099399088,5.12249356166,5.46399313243,5.80549270321,6.14699227399,6.48849184477,6.82999141554,7.17149098632,7.5129905571,7.85449012787,8.19598969865,8.53748926943,8.8789888402,9.22048841098,9.56198798176,9.90348755254};
    const double kappa_spline[30]={-21.6395565688,-21.5641675629,-21.5225112028,-21.5130514508,-21.5342522691,-21.5845293039,-21.6581396414,-21.7420392948,-21.8221380683,-21.8837908896,-21.9167553997,-21.9200173678,-21.8938574675,-21.8414464728,-21.7684762963,-21.6796222358,-21.5784701374,-21.4679438133,-21.3503236936,-21.2277666787,-21.1017425964,-20.9733966978,-20.8437244283,-20.7135746917,-20.583135408,-20.4523507819,-20.3215504736,-20.1917429161,-20.0629513946,-19.9343540344};


    T = log(T);

    if (T < TK_spline[0]) { // Below 1 K, just use that value
        ans = kappa_spline[0];
    } else if (T > TK_spline[KAPPA_10_pH_NPTS_Spline-1]) {
        // Power law extrapolation
        ans  = kappa_spline[KAPPA_10_pH_NPTS_Spline-1] + ((kappa_spline[KAPPA_10_pH_NPTS_Spline-1] - kappa_spline[KAPPA_10_pH_NPTS_Spline-2]) / (TK_spline[KAPPA_10_pH_NPTS_Spline-1] - TK_spline[KAPPA_10_pH_NPTS_Spline-2]) * (T-TK_spline[KAPPA_10_pH_NPTS_Spline-1]));
    } else { // Do spline

        TK_spline_int = (int)floor((T - TK_spline[0])*inv_BinWidth_pH);

        ans = kappa_spline[TK_spline_int] + ( T - (TK_spline[0] + BinWidth_pH*(double)TK_spline_int))*( kappa_spline[TK_spline_int+1] - kappa_spline[TK_spline_int] )*inv_BinWidth_pH;
    }
    ans = exp(ans);
    return ans;
}
__device__ double kappa_10_elec(double T, int flag)
{

    double ans;
    int TK_spline_int;

    const double BinWidth_elec=0.396997429827;
    const double inv_BinWidth_elec=1./0.396997429827;
    const double TK_spline[30]={0.0,0.396997429827,0.793994859653,1.19099228948,1.58798971931,1.98498714913,2.38198457896,2.77898200879,3.17597943861,3.57297686844,3.96997429827,4.36697172809,4.76396915792,5.16096658775,5.55796401757,5.9549614474,6.35195887722,6.74895630705,7.14595373688,7.5429511667,7.93994859653,8.33694602636,8.73394345618,9.13094088601,9.52793831584,9.92493574566,10.3219331755,10.7189306053,11.1159280351,11.512925465};
    const double kappa_spline[30]={-22.1549007191,-21.9576919899,-21.760758435,-21.5641795674,-21.3680349001,-21.1724124486,-20.9774403051,-20.78327367,-20.5901042551,-20.3981934669,-20.2078762485,-20.0195787458,-19.8339587914,-19.6518934427,-19.4745894649,-19.3043925781,-19.1444129787,-18.9986014565,-18.8720602784,-18.768679825,-18.6909581885,-18.6387511068,-18.6093755705,-18.5992098958,-18.6050625357,-18.6319366207,-18.7017996535,-18.8477153986,-19.0813436512,-19.408859606};


    T = log(T);

    if (T < TK_spline[0]) { // Below 1 K, just use that value
        ans = kappa_spline[0];
    } else if (T > TK_spline[KAPPA_10_elec_NPTS_Spline-1]) {
        // Power law extrapolation
        ans  = kappa_spline[KAPPA_10_elec_NPTS_Spline-1] + ((kappa_spline[KAPPA_10_elec_NPTS_Spline-1] - kappa_spline[KAPPA_10_elec_NPTS_Spline-2]) / (TK_spline[KAPPA_10_elec_NPTS_Spline-1] - TK_spline[KAPPA_10_elec_NPTS_Spline-2]) * (T-TK_spline[KAPPA_10_elec_NPTS_Spline-1]));

    } else { // Do spline

        TK_spline_int = (int)floor((T - TK_spline[0])*inv_BinWidth_elec);

        ans = kappa_spline[TK_spline_int] + ( T - ( TK_spline[0] + BinWidth_elec*(float)TK_spline_int ) )*( kappa_spline[TK_spline_int+1] - kappa_spline[TK_spline_int] )*inv_BinWidth_elec;
    }
    return exp(ans);
}
__device__ int find_nearest_point(double min, double max, int n, double value){
    int pos=0;
    double dn = (max - min)/(n-1);
    if (value<=(min+dn)) pos=0;              // ensures we are in the first point
    else if (value>=max) pos = n-2;          // ensures we cannot exceed the maximum point
    else pos = (int)floor((value - min)/dn); // round it down to ensure we are always either side of the cell boundary
    return pos;
}
__device__ int find_xyz_pos(int xpos, int ypos, int zpos, int len_yarr, int len_zarr){
    int pxyz = xpos*len_yarr*len_zarr + ypos*len_zarr + zpos;
    return pxyz;
}
__device__ double interpolate_heating_efficiencies(double tk, double ts, double taugp, double *arrE) {
    tk = log10(tk);
    // Check that the value doesn't exceed the bound of the table. If it does, set it to the edge value (do not interpolate)
    if(tk < Tk_min) {
        tk = Tk_min;
    }
    if(tk > Tk_max) {
        tk = Tk_max;
    }

    ts = log10(ts);
    if(ts < Ts_min) {
        ts = Ts_min;
    }
    if(ts > Ts_max) {
        ts = Ts_max;
    }

    taugp = log10(taugp);
    if(taugp < taugp_min) {
        taugp = taugp_min;
    }
    if(taugp > taugp_max) {
        taugp = taugp_max;
    }

    int itk, its, itaugp, idec;
    itk = find_nearest_point(Tk_min, Tk_max, nT, tk);
    its = find_nearest_point(Ts_min, Ts_max, nT, ts);
    itaugp = find_nearest_point(taugp_min, taugp_max, ngp, taugp);

    idec = find_xyz_pos(itk, its, itaugp, nT, ngp);

    double x0,x1,y0,y1,z0,z1,xd, yd, zd;
    double c000, c100, c001, c101, c010, c110,c011,c111;
    double c00,c01,c10,c11,c0,c1,c;

    x0 =  Tk_min + itk*(Tk_max - Tk_min)/(nT-1);        // Making these (nT-1) ensures we reach the correct edge value
    x1 =  Tk_min + (itk+1)*(Tk_max - Tk_min)/(nT-1);

    y0 =  Ts_min + its*(Ts_max - Ts_min)/(nT-1);
    y1 =  Ts_min + (its+1)*(Ts_max - Ts_min)/(nT-1);

    z0 = taugp_min + itaugp*(taugp_max - taugp_min)/(ngp-1);
    z1 = taugp_min + (itaugp+1)*(taugp_max - taugp_min)/(ngp-1);

    xd = (tk - x0)/(x1 - x0);       // Above corrections ensure this remains in [0,1]
    yd = (ts - y0)/(y1 - y0);
    zd = (taugp - z0)/(z1 - z0);

    c000 = arrE [find_xyz_pos(itk, its, itaugp, nT, ngp)];
    c100 = arrE [find_xyz_pos(itk+1, its, itaugp, nT, ngp)];
    c001 = arrE [find_xyz_pos(itk, its, itaugp+1, nT, ngp)];
    c101 = arrE [find_xyz_pos(itk+1, its, itaugp+1, nT, ngp)];
    c010 = arrE [find_xyz_pos(itk, its+1, itaugp, nT, ngp)];
    c110 = arrE [find_xyz_pos(itk+1, its+1, itaugp, nT, ngp)];
    c011 = arrE [find_xyz_pos(itk, its+1, itaugp+1, nT, ngp)];
    c111 = arrE [find_xyz_pos(itk+1, its+1, itaugp+1, nT, ngp)];

    c00 = c000*(1.-xd) + c100*xd;
    c01 = c001*(1.-xd) + c101*xd;
    c10 = c010*(1.-xd) + c110*xd;
    c11 = c011*(1.-xd) + c111*xd;

    c0 = c00*(1.-yd) + c10*yd;
    c1 = c01*(1.-yd) + c11*yd;

    c = c0*(1.-zd) + c1*zd;
    return c;
}
__device__ double alpha_A(double T){
    double logT, ans;
    logT = log(T/(double)1.1604505e4);
    const double l2 = logT * logT;
    const double l3 = l2 * logT;
    const double l4 = l2 * l2;
    const double l5 = l4 * logT;
    const double l6 = l4 * l2;
    const double l7 = l6 * logT;
    const double l8 = l4 * l4;
    const double l9 = l8 * logT;
    ans = pow(E, -28.6130338 - 0.72411256*logT - 2.02604473e-2*l2
              - 2.38086188e-3*l3 - 3.21260521e-4*l4
              - 1.42150291e-5*l5 + 4.98910892e-6*l6
              + 5.75561414e-7*l7 - 1.85676704e-8*l8
              - 3.07113524e-9 * l9);
    return ans;
}
__device__ double Energy_Lya_heating(double Tk,double Ts,double tau,int flag,const double *tc,const double *ti){return interpolate_heating_efficiencies(Tk,Ts,tau,(double*)(flag==2?tc:ti));}
__device__ void Energy_Lya_heating_pair(double tk, double ts, double taugp, const double * __restrict__ tc, const double * __restrict__ ti, double *out_c, double *out_i) {
    tk = log10(tk);
    if (tk < Tk_min) tk = Tk_min;
    if (tk > Tk_max) tk = Tk_max;
    ts = log10(ts);
    if (ts < Ts_min) ts = Ts_min;
    if (ts > Ts_max) ts = Ts_max;
    taugp = log10(taugp);
    if (taugp < taugp_min) taugp = taugp_min;
    if (taugp > taugp_max) taugp = taugp_max;

    const int itk = find_nearest_point(Tk_min, Tk_max, nT, tk);
    const int its = find_nearest_point(Ts_min, Ts_max, nT, ts);
    const int itaugp = find_nearest_point(taugp_min, taugp_max, ngp, taugp);

    const double x0 = Tk_min + itk*(Tk_max - Tk_min)/(nT-1);
    const double x1 = Tk_min + (itk+1)*(Tk_max - Tk_min)/(nT-1);
    const double y0 = Ts_min + its*(Ts_max - Ts_min)/(nT-1);
    const double y1 = Ts_min + (its+1)*(Ts_max - Ts_min)/(nT-1);
    const double z0 = taugp_min + itaugp*(taugp_max - taugp_min)/(ngp-1);
    const double z1 = taugp_min + (itaugp+1)*(taugp_max - taugp_min)/(ngp-1);

    const double xd = (tk - x0)/(x1 - x0);
    const double yd = (ts - y0)/(y1 - y0);
    const double zd = (taugp - z0)/(z1 - z0);

    const int p000 = find_xyz_pos(itk, its, itaugp, nT, ngp);
    const int p100 = find_xyz_pos(itk+1, its, itaugp, nT, ngp);
    const int p001 = find_xyz_pos(itk, its, itaugp+1, nT, ngp);
    const int p101 = find_xyz_pos(itk+1, its, itaugp+1, nT, ngp);
    const int p010 = find_xyz_pos(itk, its+1, itaugp, nT, ngp);
    const int p110 = find_xyz_pos(itk+1, its+1, itaugp, nT, ngp);
    const int p011 = find_xyz_pos(itk, its+1, itaugp+1, nT, ngp);
    const int p111 = find_xyz_pos(itk+1, its+1, itaugp+1, nT, ngp);

    {
        const double c00 = tc[p000]*(1.-xd) + tc[p100]*xd;
        const double c01 = tc[p001]*(1.-xd) + tc[p101]*xd;
        const double c10 = tc[p010]*(1.-xd) + tc[p110]*xd;
        const double c11 = tc[p011]*(1.-xd) + tc[p111]*xd;
        const double c0 = c00*(1.-yd) + c10*yd;
        const double c1 = c01*(1.-yd) + c11*yd;
        *out_c = c0*(1.-zd) + c1*zd;
    }
    {
        const double c00 = ti[p000]*(1.-xd) + ti[p100]*xd;
        const double c01 = ti[p001]*(1.-xd) + ti[p101]*xd;
        const double c10 = ti[p010]*(1.-xd) + ti[p110]*xd;
        const double c11 = ti[p011]*(1.-xd) + ti[p111]*xd;
        const double c0 = c00*(1.-yd) + c10*yd;
        const double c1 = c01*(1.-yd) + c11*yd;
        *out_i = c0*(1.-zd) + c1*zd;
    }
}

/* One iteration of the existing Wouthuysen–Field fixed-point map. */
__device__ inline double spin_next(
    double spin, double inv_t, double inv_t_sq, double coupling,
    double cmb, double inv_radiation, double collisions
) {
    const double inv_s = 1.0 / spin;
    const double alpha =
        (1.0 - 0.0631789*inv_t + 0.115995*inv_t_sq -
         0.401403*inv_t*inv_s + 0.336463*inv_t_sq*inv_s) * coupling;
    return (cmb+alpha+collisions) /
        (cmb*inv_radiation + alpha*(inv_t + 0.405535*inv_t*inv_s -
            0.405535*inv_t_sq) + collisions*inv_t);
}

extern "C" __global__ void update(const float *density0,const float *prev_x,const float *prev_t,const float *prev_ts,float *out_x,float *out_t,float *out_ts,double *dxheat_dt_box,double *dxion_source_dt_box,double *dxlya_dt_box,double *dstarlya_dt_box,double *dstarlya_cont_dt_box,double *dstarlya_inj_dt_box,const double *sc,const double *tablec,const double *tablei,long n,int use_cmb,int use_lya,int *bad, unsigned int *relaxed){long i=(long)blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;double curr_delNL0,x_e,T,prev_Ts,tau21,xCMB,dxion_sink_dt,dxe_dzp,dadia_dzp,dspec_dzp,dcomp_dzp,dxheat_dzp,dCMBheat_dzp,eps_CMB,E_continuum,E_injected,Ndot_alpha_cont,Ndot_alpha_inj,eps_Lya_cont,eps_Lya_inj,J_alpha_tot,T_inv,T_inv_sq,xc_fast,xi_power,xa_tilde_fast_arg,TS_fast,TSold_fast,xa_tilde_fast;double zp=sc[0];
double dzp=sc[1];
double growth_factor_zp=sc[2];
double dgrowth_factor_dzp=sc[3];
double dt_dzp=sc[4];
double const_zp_prefactor=sc[5];
double prefactor_1=sc[6];
double prefactor_2=sc[7];
double dcomp_dzp_prefactor=sc[8];
double Trad_fast=sc[9];
double xc_inverse=sc[10];
double TS_prefactor=sc[11];
double xa_tilde_prefactor=sc[12];
double Trad_fast_inv=sc[13];
double No=sc[14];
double N_b0=sc[15];
double f_H=sc[16];
double f_He=sc[17];
double hub_z=sc[18];
double clumping=sc[19];
double tstar=sc[20];
double crit=sc[21];
double min_density_limit=sc[22];
double high_inv=sc[23];
                            curr_delNL0 = density0[i];

                            x_e = prev_x[i];
                            T = prev_t[i];

                            // add prefactors
                            dxheat_dt_box[i] *= const_zp_prefactor;
                            dxion_source_dt_box[i] *= const_zp_prefactor;

                            dxlya_dt_box[i] *= const_zp_prefactor*prefactor_1 * (1.+curr_delNL0*growth_factor_zp);
                            dstarlya_dt_box[i] *= prefactor_2;

                            //Include pre-factors in Lya flux for Lya heating
                            if (use_lya){
                                dstarlya_cont_dt_box[i] *= prefactor_2;
                                dstarlya_inj_dt_box[i] *= prefactor_2;
                            }

                            

                            //Added calculations of xCMB and tau_21 [eq. (2) and (4) of Reis et al. 2021]
                            const double zp1 = 1.0 + zp;
                            const double zp1_3 = zp1 * zp1 * zp1;
                            prev_Ts = prev_ts[i];
                            tau21 = (3*hplank*A10_HYPERFINE*C*Lambda_21*Lambda_21/32./PI/k_B) * ((1-x_e)*No*zp1_3) /prev_Ts/hub_z;
                            xCMB = (tau21 == 0.0) ? 1.0 : -expm1(-tau21)/tau21;

                            // Now we can solve the evolution equations  //

                            // First let's do dxe_dzp //
                            dxion_sink_dt = alpha_A(T) * clumping * x_e*x_e * f_H * prefactor_1 * \
                                            (1.+curr_delNL0*growth_factor_zp);
                            
                                dxe_dzp = dt_dzp*(dxion_source_dt_box[i] - dxion_sink_dt );
                            

                            // Next, let's get the temperature components //
                            // first, adiabatic term
                            dadia_dzp = 3/(1.0+zp);
                            if (fabs(curr_delNL0) > FRACT_FLOAT_ERR) // add adiabatic heating/cooling from structure formation
                                dadia_dzp += dgrowth_factor_dzp/(1.0/curr_delNL0+growth_factor_zp);

                            dadia_dzp *= (2.0/3.0)*T;

                            // next heating due to the changing species
                            dspec_dzp = - dxe_dzp * T / (1+x_e);

                            // next, Compton heating
                            //                dcomp_dzp = dT_comp(zp, T, x_e);
                            dcomp_dzp = dcomp_dzp_prefactor*(x_e/(1.0+x_e+f_He))*( Trad_fast - T );

                            // next, X-ray heating
                            dxheat_dzp = dxheat_dt_box[i] * dt_dzp * 2.0 / 3.0 / k_B / (1.0+x_e);
                            

                            //next, CMB heating rate
                            dCMBheat_dzp = 0.;
                            if (use_cmb) {
                                //Meiksin et al. 2021
                                eps_CMB = (3./4.) * (T_cmb*(1.+zp)/T21) * A10_HYPERFINE * f_H * (hplank*hplank/Lambda_21/Lambda_21/m_p) * (1.+2.*T/T21);
                                dCMBheat_dzp = 	-eps_CMB * (2./3./k_B/(1.+x_e))/hub_z/(1.+zp);
                            }
                            //lastly, Ly-alpha heating rate
                            eps_Lya_cont = 0.;
                            eps_Lya_inj = 0.;


                            if (use_lya) {
                                const double opd = 1.0 + curr_delNL0*growth_factor_zp;
                                const double lya_tau = 1.342881e-7 / hub_z*No*zp1_3 * opd*(1.0-x_e);
                                Energy_Lya_heating_pair(T, prev_ts[i], lya_tau, tablec, tablei, &E_continuum, &E_injected);
                                if (isnan(E_continuum) || isinf(E_continuum)){
                                    E_continuum = 0.;
                                }
                                if (isnan(E_injected) || isinf(E_injected)){
                                    E_injected = 0.;
                                }
                                const double lya_norm = (4.*PI*Ly_alpha_HZ) / (N_b0*zp1_3*opd)/zp1/C;
                                Ndot_alpha_cont = lya_norm * dstarlya_cont_dt_box[i];
                                Ndot_alpha_inj = lya_norm * dstarlya_inj_dt_box[i];
                                eps_Lya_cont = - Ndot_alpha_cont * E_continuum * (2. / 3. /k_B/ (1.+x_e));
                                eps_Lya_inj = - Ndot_alpha_inj * E_injected * (2. / 3. /k_B/ (1.+x_e));
                                
                            }

                            //update quantities
                            x_e += ( dxe_dzp ) * dzp; // remember dzp is negative
                            if (x_e > 1) // can do this late in evolution if dzp is too large
                                x_e = 1 - FRACT_FLOAT_ERR;
                            else if (x_e < 0)
                                x_e = 0;

                            //Add CMB and Lya heating rates, and evolve
                            if (T < MAX_TK) {
                                
                                    T += ( dxheat_dzp + dcomp_dzp + dspec_dzp + dadia_dzp + dCMBheat_dzp + eps_Lya_cont + eps_Lya_inj) * dzp;
                                

                            }

                            if(!isfinite(T)){atomicCAS(bad,0,101);return;}
                            if (T<0){ // spurious bahaviour of the trapazoidalintegrator. generally overcooling in underdensities
                                T = T_cmb*(1+zp);
                            }

                            out_x[i] = x_e;
                            out_t[i] = T;

                            J_alpha_tot = ( dxlya_dt_box[i] + dstarlya_dt_box[i] ); //not really d/dz, but the lya flux
                            

                            // Note: to make the code run faster, the get_Ts function call to evaluate the spin temperature was replaced with the code below.
                            // Algorithm is the same, but written to be more computationally efficient
                            // Added corrections from xCMB [eq. (3) of Reis et al. 2021]
                            const float logf_T = glibc_logf((float)T);
                            T_inv = (double)glibc_expf((float)((-1.0) * (double)logf_T));
                            T_inv_sq = (double)glibc_expf((float)((-2.0) * (double)logf_T));

                            xc_fast = (1.0+curr_delNL0*growth_factor_zp)*xc_inverse*\
                                    ( (1.0-x_e)*No*kappa_10(T,0) + x_e*N_b0*kappa_10_elec(T,0) + x_e*No*kappa_10_pH(T,0) );

                            xi_power = TS_prefactor * cbrt((1.0+curr_delNL0*growth_factor_zp)*(1.0-x_e)*T_inv_sq);

                            
                                xa_tilde_fast_arg = xa_tilde_prefactor*J_alpha_tot / ( 1.0 + 2.98394*xi_power + 1.53583*xi_power*xi_power + 3.85289*xi_power*xi_power*xi_power );
                            

                            //if (J_alpha_tot > 1.0e-20) { // Must use WF effect
                            // New in v1.4
                            if (fabs(J_alpha_tot) > 1.0e-20) { // Must use WF effect
                                TS_fast = Trad_fast;
                                TSold_fast = 0.0;
                                int iterations = 0;
                                for (;;) {
                                    const double residual = fabs(TS_fast-TSold_fast)/TS_fast;
#ifdef PS_ENABLE_SPIN_RELAX
                                    const bool near_cutoff = fabs(residual - 1.0e-3) <= 2.0e-7;
#else
                                    const bool near_cutoff = false;
#endif
                                    if (!near_cutoff && !(residual > 1.0e-3)) break;
                                    if (++iterations > 10000) { atomicCAS(bad,0,2); return; }

                                    const double trial = spin_next(TS_fast,T_inv,T_inv_sq,
                                        xa_tilde_fast_arg,xCMB,Trad_fast_inv,xc_fast);
                                    if (near_cutoff) {
                                        // Dampen the next step only when the local map strongly contracts.
                                        // Confirm the returned temperature still meets the original criterion.
                                        const double previous_step = fabs(TS_fast - TSold_fast);
                                        const double midpoint = TS_fast + 0.5*(trial - TS_fast);
                                        if (isfinite(midpoint) && midpoint > 0.0 &&
                                            fabs(trial - TS_fast) <= 0.18*previous_step &&
                                            fabs(spin_next(midpoint,T_inv,T_inv_sq,xa_tilde_fast_arg,
                                                xCMB,Trad_fast_inv,xc_fast)-midpoint)/midpoint <= 1.0e-3) {
                                            TS_fast = midpoint;
                                            atomicAdd(relaxed,1);
                                            break;
                                        }
                                        atomicCAS(bad,0,3);
                                        if (!(residual > 1.0e-3)) break;
                                    }
                                    TSold_fast = TS_fast;
                                    TS_fast = trial;
                                }
                            } else { // Collisions only
                                TS_fast = (xCMB + xc_fast)/(xCMB*Trad_fast_inv + xc_fast*T_inv);

                                xa_tilde_fast = 0.0;
                            }

                            if(TS_fast < 0.) {
                                // It can very rarely result in a negative spin temperature. If negative, it is a very small number.
                                //Take the absolute value, the optical depth can deal with very large numbers, so ok to be small
                                TS_fast = fabs(TS_fast);
                            }

                            out_ts[i] = TS_fast;

}
