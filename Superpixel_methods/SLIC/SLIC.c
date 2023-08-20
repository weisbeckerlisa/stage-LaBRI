//SuperPixel.c


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#include "mex.h"

#ifndef max
#define max(a, b) ((a)>(b)?(a):(b))
#endif

#ifndef min
#define min(a, b) ((a)<(b)?(a):(b))
#endif

void RGB2XYZ(
        const float		sR,
        const float		sG,
        const float		sB,
        double*			X,
        double*			Y,
        double*			Z)
{
//     double R = sR/255.0;
//     double G = sG/255.0;
//     double B = sB/255.0;
    
        double R = sR;
    double G = sG;
    double B = sB;
    
    double r, g, b;
    
    if (R <= 0.04045)
        r = R/12.92;
    else
        r = pow((R+0.055)/1.055,2.4);
    
    if (G <= 0.04045)
        g = G/12.92;
    else
        g = pow((G+0.055)/1.055,2.4);
    
    if (B <= 0.04045)
        b = B/12.92;
    else
        b = pow((B+0.055)/1.055,2.4);
    
    
    *X = r*0.4124564 + g*0.3575761 + b*0.1804375;
    *Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
    *Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
    
}



void RGB2LAB(const float sR, const float sG, const float sB, double* lval, double* aval, double* bval)
{
    //------------------------
    // sRGB to XYZ conversion
    //------------------------
    double X, Y, Z;
    RGB2XYZ(sR, sG, sB, &X, &Y, &Z);
    
    //------------------------
    // XYZ to LAB conversion
    //------------------------
    double epsilon = 0.008856;	//actual CIE standard
    double kappa   = 903.3;		//actual CIE standard
    
    double Xr = 0.950456;	//reference white
    double Yr = 1.0;		//reference white
    double Zr = 1.088754;	//reference white
    
    double xr = X/Xr;
    double yr = Y/Yr;
    double zr = Z/Zr;
    
    double fx, fy, fz;
    if(xr > epsilon)
        fx = pow(xr, 1.0/3.0);
    else
        fx = (kappa*xr + 16.0)/116.0;
    
    if(yr > epsilon)
        fy = pow(yr, 1.0/3.0);
    else
        fy = (kappa*yr + 16.0)/116.0;
    
    if(zr > epsilon)
        fz = pow(zr, 1.0/3.0);
    else
        fz = (kappa*zr + 16.0)/116.0;
    
    *lval = 116.0*fy-16.0;
    *aval = 500.0*(fx-fy);
    *bval = 200.0*(fy-fz);
}


void DoRGBtoLABConversion(
        float*      ubuffr,
        float*       ubuffg,
        float*       ubuffb,
        double*      lvec,
        double*      avec,
        double*      bvec,
        const int   m_height,
        const int   m_width)
{
    
    int sz = m_width*m_height;

        for( int j = 0; j < sz; j++ )
        {
            float r = ubuffr[j];
            float g = ubuffg[j];
            float b = ubuffb[j];
            
            RGB2LAB(r, g, b, &lvec[j], &avec[j], &bvec[j]);
        }
    
}


void DetectLabEdges(
	const double*				lvec,
	const double*				avec,
	const double*				bvec,
	const int					height,
	const int					width,
	double*     				edges)
{
	for( int j = 1; j < height-1; j++ )
	{
		for( int k = 1; k < width-1; k++ )
		{
			int i = j+k*height;

			double dx = (lvec[i-1]-lvec[i+1])*(lvec[i-1]-lvec[i+1]) +
						(avec[i-1]-avec[i+1])*(avec[i-1]-avec[i+1]) +
						(bvec[i-1]-bvec[i+1])*(bvec[i-1]-bvec[i+1]);

			double dy = (lvec[i-width]-lvec[i+width])*(lvec[i-width]-lvec[i+width]) +
						(avec[i-width]-avec[i+width])*(avec[i-width]-avec[i+width]) +
						(bvec[i-width]-bvec[i+width])*(bvec[i-width]-bvec[i+width]);

			edges[i] = dx*dx + dy*dy;
		}
	}
}






//===========================================================================
///	PerturbSeeds
//===========================================================================
void PerturbSeeds(
        double *m_lvec, 
        double* m_avec,
        double * m_bvec,        
	double*				kseedsl,
	double*				kseedsa,
	double*				kseedsb,
	double*					kseedsx,
	double*					kseedsy,
    double*	                  edges,
        const int m_height,
        const int m_width,
        const int numseeds)
{
	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	for( int n = 0; n < numseeds; n++ )
	{
		int ox = kseedsx[n];//original x
		int oy = kseedsy[n];//original y
		int oind = ox*m_height + oy;

		int storeind = oind;
		for( int i = 0; i < 8; i++ )
		{
			int nx = ox+dx8[i];//new x
			int ny = oy+dy8[i];//new y

			if( nx >= 0 && nx < m_width && ny >= 0 && ny < m_height)
			{
				int nind = nx*m_height + ny;
				if( edges[nind] < edges[storeind])
				{
					storeind = nind;
				}
			}
		}
		if(storeind != oind)
		{
			kseedsx[n] = storeind%m_width;
			kseedsy[n] = storeind/m_width;
			kseedsl[n] = m_lvec[storeind];
			kseedsa[n] = m_avec[storeind];
			kseedsb[n] = m_bvec[storeind];
		}
	}
}


void GetLABXYSeeds_ForGivenStepSize(
        double *               m_lvec,
        double *               m_avec,
        double *               m_bvec,
        double *        		kseedsl,
        double *				kseedsa,
        double *				kseedsb,
        double *				kseedsx,
        double *				kseedsy,
        const int				STEP,
        const int               m_height,
        const int               m_width,
        const int               xstrips,
        const int               ystrips,
        const double            xerrperstrip,
        const double            yerrperstrip,
        const int				perturbseeds,
        double *                edgemap)
{
    const int hexgrid = 0;
	int numseeds = 0;
	int n = 0;

	int xoff = STEP/2;
	int yoff = STEP/2;
	//-------------------------
	numseeds = xstrips*ystrips;
	//-------------------------

	for( int y = 0; y < ystrips; y++ )
	{
		int ye = y*yerrperstrip;
		for( int x = 0; x < xstrips; x++ )
		{
			int xe = x*xerrperstrip;
            int seedx = (x*STEP+xoff+xe);
            
            if (hexgrid){ 
                seedx = x*STEP+(xoff<<(y&0x1))+xe; 
                seedx = min(m_width-1,seedx); 
            }//for hex grid sampling
            
            int seedy = (y*STEP+yoff+ye);
            int i = seedx*m_height + seedy;
			
			kseedsl[n] = m_lvec[i];
			kseedsa[n] = m_avec[i];
			kseedsb[n] = m_bvec[i];
            kseedsx[n] = seedx;
            kseedsy[n] = seedy;
			n++;
		}
	}

	
	if(perturbseeds)
	{
		PerturbSeeds(m_lvec, m_avec, m_bvec, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, edgemap, m_height, m_width, numseeds);
	}
    
}




void PerformSuperpixelSLIC(
        double * m_lvec, 
        double * m_avec, 
        double * m_bvec,
	double *				kseedsl,
	double *				kseedsa,
	double *				kseedsb,
	double *				kseedsx,
	double *				kseedsy,
    int *					klabels,
        const int				STEP,
    double *                  edgemap,
	const double				M,
        const int numseeds,
        const int m_height,
        const int m_width,
        int nbr_iter)
{
	int sz = m_width*m_height;
	const int numk = numseeds;
	//----------------
	int offset = STEP;
        //if(STEP < 8) offset = STEP*1.5;//to prevent a crash due to a very small step size
	//----------------
    
    
    double * clustersize = (double *)calloc(numk, sizeof(double));
    double * inv = (double *)calloc(numk, sizeof(double));//to store 1/clustersize[k] values
    
    double * sigmal = (double *)calloc(numk, sizeof(double));
    double * sigmaa = (double *)calloc(numk, sizeof(double));
    double * sigmab = (double *)calloc(numk, sizeof(double));
    double * sigmax = (double *)calloc(numk, sizeof(double));
    double * sigmay = (double *)calloc(numk, sizeof(double));
    
    double * distvec = (double *)calloc(sz, sizeof(double));
       
    int pw = 0;

	double invwt = 1.0/((STEP/M)*(STEP/M));
    
//     printf("offset %d\n", offset);
//     mexEvalString("drawnow");

	int x1, y1, x2, y2;
	double l, a, b;
	double dist;
	double distxy;
	for( int itr = 0; itr < nbr_iter; itr++ )
	{
         for (int i=0; i<sz; i++) {
            distvec[i] = DBL_MAX;
        }
         
		for( int n = 0; n < numk; n++ )
		{
                        y1 = max(0.0,			kseedsy[n]-offset);
                        y2 = min((double)m_height,	kseedsy[n]+offset);
                        x1 = max(0.0,			kseedsx[n]-offset);
                        x2 = min((double)m_width,	kseedsx[n]+offset);


			for( int y = y1; y < y2; y++ )
			{
				for( int x = x1; x < x2; x++ )
				{
					int i = x*m_height + y;

					l = m_lvec[i];
					a = m_avec[i];
					b = m_bvec[i];

// 					dist =			(l - kseedsl[n])*(l - kseedsl[n]) +
// 									(a - kseedsa[n])*(a - kseedsa[n]) +
// 									(b - kseedsb[n])*(b - kseedsb[n]);
                    
                    dist = 0;
                for (int dx=-pw; dx<=pw; dx++) {
                    for (int dy=-pw; dy<=pw; dy++) {
                        if ((x+dx<m_width)&&(x+dx>=0)&(y+dy>=0)&(y+dy<m_height)){
                            int pos = y+dy + (x+dx)*m_height;
                            dist += (m_lvec[pos] - kseedsl[n])*(m_lvec[pos] - kseedsl[n]) +
                                    (m_avec[pos] - kseedsa[n])*(m_avec[pos] - kseedsa[n]) +
                                    (m_bvec[pos] - kseedsb[n])*(m_bvec[pos] - kseedsb[n]);
                        }
                    }                    
                }
               dist /= (2*pw+1)*(2*pw+1);
               

					distxy =		(x - kseedsx[n])*(x - kseedsx[n]) +
									(y - kseedsy[n])*(y - kseedsy[n]);
					
					//------------------------------------------------------------------------
 					dist += distxy*invwt;  //dist = sqrt(dist) + sqrt(distxy*invwt);//this is more exact
//                      dist = distxy*invwt;
					//------------------------------------------------------------------------
					if( dist < distvec[i] )
					{
						distvec[i] = dist;
						klabels[i]  = n;
					}
				}
			}
		}
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		//instead of reassigning memory on each iteration, just reset.
	for(int i=0; i<numk; i++) {
        sigmal[i] = (double) 0;
            sigmaa[i] = (double) 0;
            sigmab[i] = (double) 0;
            sigmax[i] = (double) 0;
            sigmay[i] = (double) 0;
            clustersize[i] = (double) 0;
    }
		//------------------------------------
		//edgesum.assign(numk, 0);
		//------------------------------------

		int ind = 0;
		for( int r = 0; r < m_width; r++ )
		{
			for( int c = 0; c < m_height; c++ )
			{
				sigmal[klabels[ind]] += m_lvec[ind];
				sigmaa[klabels[ind]] += m_avec[ind];
				sigmab[klabels[ind]] += m_bvec[ind];
				sigmax[klabels[ind]] += r;
				sigmay[klabels[ind]] += c;
				//------------------------------------
				//edgesum[klabels[ind]] += edgemag[ind];
				//------------------------------------
				clustersize[klabels[ind]] += 1.0;
				ind++;
			}
		}

		for( int k = 0; k < numk; k++ )
		{
			if( clustersize[k] <= 0 ) clustersize[k] = 1;
			inv[k] = 1.0/clustersize[k];//computing inverse now to multiply, than divide later
		}
		
		for( int k = 0; k < numk; k++ )
		{
			kseedsl[k] = sigmal[k]*inv[k];
			kseedsa[k] = sigmaa[k]*inv[k];
			kseedsb[k] = sigmab[k]*inv[k];
			kseedsx[k] = sigmax[k]*inv[k];
			kseedsy[k] = sigmay[k]*inv[k];
			//------------------------------------
			//edgesum[k] *= inv[k];
			//------------------------------------
		}
	}
    
    
    free(clustersize);
    free(inv);
    free(distvec);
    free(sigmal);
    free(sigmaa);
    free(sigmab);
    free(sigmax);
    free(sigmay);
    
}



void EnforceLabelConnectivity(
	const int*					labels,//input labels that need to be corrected to remove stray labels
	const int					height,
	const int					width,
	int*						nlabels,//new labels
	int*						numlabels,//the number of labels changes in the end if segments are removed
	const int					K) //the number of superpixels desired by the user
{
//	const int dx8[8] = {-1, -1,  0,  1, 1, 1, 0, -1};
//	const int dy8[8] = { 0, -1, -1, -1, 0, 1, 1,  1};

	const int dx4[4] = {-1,  0,  1,  0};
	const int dy4[4] = { 0, -1,  0,  1};

	const int sz = width*height;
	const int SUPSZ = sz/K;
	//nlabels.resize(sz, -1);
	for( int i = 0; i < sz; i++ ) 
        nlabels[i] = -1;
    
	int label = 0;
    
    int * xvec= (int *) calloc(sz, sizeof(int));
    int * yvec= (int *) calloc(sz, sizeof(int));    
    
	int oindex = 0;
	int adjlabel = 0;//adjacent label
	for( int j = 0; j < width; j++ )
	{
		for( int k = 0; k < height; k++ )
		{
			if( 0 > nlabels[oindex] )
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = j;
				yvec[0] = k;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				for( int n = 0; n < 4; n++ )
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if( (x >= 0 && x < width) && (y >= 0 && y < height) )
					{
						int nindex = x*height + y;
						if(nlabels[nindex] >= 0) 
                            adjlabel = nlabels[nindex];
					}
				}

				int count = 1;
				for( int c = 0; c < count; c++ )
				{
					for( int n = 0; n < 4; n++ )
					{
						int x = xvec[c] + dx4[n];
						int y = yvec[c] + dy4[n];

						if( (x >= 0 && x < width) && (y >= 0 && y < height) )
						{
							int nindex = x*height + y;

							if( 0 > nlabels[nindex] && labels[oindex] == labels[nindex] )
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}

					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before,
                //and decrement label count.
				//-------------------------------------------------------
				if(count <= SUPSZ >> 2)
				{
					for( int c = 0; c < count; c++ )
					{
						int ind = yvec[c]+xvec[c]*height;
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	*numlabels = label;
    
//     
//     free(xvec);
//     free(yvec);
}





void DoSuperpixelSegmentation_ForGivenSuperpixelSize(
        float*                      ubuffr,
        float*                      ubuffg,
        float*                      ubuffb,
        const int					height,
        const int					width,
        int*						klabels,
        int                         numlabels,
        const int                   superpixelsize,
        const double                compactness,
        const int                   perturbseeds, 
        int nbr_iter)
{
    
    //------------------------------------------------
    const int STEP = sqrt((double)superpixelsize)+0.5;
    int m_width  = width;
    int m_height = height;
    int sz = m_width*m_height;
    
    double* m_lvec = (double *)calloc(sz, sizeof(double));
    double* m_avec = (double *)calloc(sz, sizeof(double));
    double* m_bvec = (double *)calloc(sz, sizeof(double));
    
    //--------------------------------------------------
    DoRGBtoLABConversion(ubuffr, ubuffg, ubuffb, m_lvec, m_avec, m_bvec, m_height, m_width);
    //--------------------------------------------------

    double * edgemap = (double *)calloc(sz, sizeof(double));
    
    //perturb seeds is not absolutely necessary, one can set this flag to false
    if(perturbseeds)
        DetectLabEdges(m_lvec, m_avec, m_bvec, m_height, m_width, edgemap);
    
    int xstrips = 0.5+(double)m_width/(double)STEP;
	int ystrips = 0.5+(double)m_height/(double)STEP;
    int numseeds = xstrips*ystrips;

    int xerr = m_width  - STEP*xstrips;if(xerr < 0){xstrips--;xerr = m_width - STEP*xstrips;}
    int yerr = m_height - STEP*ystrips;if(yerr < 0){ystrips--;yerr = m_height- STEP*ystrips;}

	double xerrperstrip = (double)xerr/(double)xstrips;
	double yerrperstrip = (double)yerr/(double)ystrips;
    
    double * kseedsl = (double *)calloc(numseeds, sizeof(double));
    double * kseedsa = (double *)calloc(numseeds, sizeof(double));
    double * kseedsb = (double *)calloc(numseeds, sizeof(double));
    double * kseedsx = (double *)calloc(numseeds, sizeof(double));
    double * kseedsy = (double *)calloc(numseeds, sizeof(double));
   
    GetLABXYSeeds_ForGivenStepSize(m_lvec, m_avec, m_bvec, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, 
            STEP, m_height, m_width,xstrips,ystrips,  xerrperstrip, yerrperstrip,perturbseeds, edgemap);

    
    
    PerformSuperpixelSLIC(m_lvec, m_avec, m_bvec, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, klabels, STEP,
            edgemap,compactness, numseeds, m_height, m_width, nbr_iter);
    numlabels = numseeds;
// 
    int * nlabels = (int *)calloc(sz, sizeof(int));
    EnforceLabelConnectivity(klabels, m_height, m_width, nlabels, &numlabels, (double)sz/((double)STEP*STEP));
//     
    for(int i = 0; i < sz; i++ )
        klabels[i] = nlabels[i];
    
    //Free
    free(m_lvec);
    free(m_avec);
    free(m_bvec);
    free(nlabels);
    free(kseedsl);
    free(kseedsa);
    free(kseedsb);
    free(kseedsx);
    free(kseedsy);
    free(edgemap);
    
}




void mexFunction(int nlhs, mxArray *plhs[],int nrhs,const mxArray *prhs[]){
    (void)nlhs;
    (void)nrhs;
    
    /* INPUTS */
    //RGB image
    float* img = (float*) mxGetPr(prhs[0]);
    
    int SP_nbr = (int) mxGetScalar(prhs[1]);
    double compactness = (double) mxGetScalar(prhs[2]);
    int nbr_iter = (int) mxGetScalar(prhs[3]);
    
    int perturbseeds = 0;
    //Dimensions stuff
    const mwSize* img_dims = mxGetDimensions(prhs[0]);
    int h_img = img_dims[0];
    int w_img = img_dims[1];
    int h_w_img = h_img*w_img;
    
    /*Tables*/
    int *klabels = (int *)calloc(h_w_img, sizeof(int));
    float *ubuffr =(float *)calloc(h_w_img, sizeof(float));
    float *ubuffg =(float *)calloc(h_w_img, sizeof(float));
    float *ubuffb =(float *)calloc(h_w_img, sizeof(float));
    
    for(int i=0; i < h_w_img; i++) {
        ubuffr[i] = img[i];
        ubuffg[i] = img[i+h_w_img];
        ubuffb[i] = img[i+2*h_w_img];
        klabels[i] = -1;
    }
    
    const int superpixelsize = 0.5+(double)w_img*h_img/(double)SP_nbr;
    DoSuperpixelSegmentation_ForGivenSuperpixelSize(ubuffr, ubuffg, ubuffb, h_img, w_img ,
            klabels,SP_nbr,superpixelsize,compactness, perturbseeds, nbr_iter);
    
    
    //Outputs
    mwSize dims[2];
    dims[0] = h_img;
    dims[1] = w_img;
    plhs[0] = mxCreateNumericArray(2, dims, mxINT32_CLASS, mxREAL);
    int * klabels_output = (int *)mxGetPr(plhs[0]);
    for (int i=0; i<h_img*w_img; i++) {
//         klabels_output[((int)(i/h_img))*h_img + i%h_img] = (int) klabels[i];
        klabels_output[i] = (int) klabels[i];
    }
    

        free(ubuffr);
    free(ubuffg);
    free(ubuffb);
    free(klabels);

    
}







