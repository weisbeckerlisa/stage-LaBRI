

#include "./utils/CImg.h"
#include <iostream>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include "./utils/SuperpixelHierarchyMex.hpp"

using namespace std;
using namespace cimg_library;

int findset(int i, int *parent)
{
    int p = parent[i];
    if (i != p)
    {
        parent[i] = findset(p, parent);
    }
    return parent[i];
}

void getSuperpixels(int *parent, int *label, int *treeu, int *treev, int &nvertex, int &nregion, int &N)
{
    if (N < 1 || N > nvertex)
    {
        printf("error");
        exit(1);
    }

    int end   = nvertex-N;
    int begin = nvertex-nregion;
    if (nregion < N)
    {
        for (int i=0; i<nvertex; ++i) parent[i] = i;
        begin = 0;
    }

    for (int i=begin; i<end; ++i)
    {
        int u  = treeu[i];
        int v  = treev[i];
        int pu = findset(u,parent);
        int pv = findset(v,parent);
        if (pu < pv)
            parent[pv] = pu;
        else
            parent[pu] = pv;
    }

    nregion = 0;
    for (int i=0; i<nvertex; ++i)
    {
        int p = findset(i,parent);
        if (i == p)
            label[i] = nregion++;
        else
            label[i] = label[p];
    }
}




int main(int argc, char* argv[])
{

    //Inputs
    string img_name = string(cimg_option("-i","","Input image file"));
    string contour_name = string(cimg_option("-c","","Input contour image file"));
    int connect = cimg_option("-w",4,"Edge_weight: balance between edge and color feature (default: 4)");
    int iterSwitch = cimg_option("-p", 4, "Compactness value (default 4)");

    //Image loading with CImg
    cout << img_name.c_str() << "\n";
    CImg<unsigned char> img_in(img_name.c_str());
    CImg<unsigned char> image = img_in;
    int w = image.width();
    int h = image.height();

    cout << contour_name.c_str() << "\n";
    CImg<unsigned char> edge_in(contour_name.c_str());
    CImg<unsigned char> edge = edge_in;
    for (int i=0; i<w*h; i++){
        edge[i] = (unsigned char) 0;
    }


    // Be careful with le sens de parcours matlab/c
    unsigned char * image_shift = (unsigned char *) calloc((h*w)*3,sizeof(unsigned char));
    unsigned char * edge_shift = (unsigned char *) calloc((h*w),sizeof(unsigned char));
    for (int i=0; i<w; i++){
        for (int j=0; j<h; j++) {
            for (int d=0; d<3; d++) {
                image_shift[j+i*h + d*h*w] = image[i+j*w + d*h*w];
            }
            edge_shift[j+i*h] = edge[i+j*w];
        }
    }

    //Computation of the Superpixel Hierarchy
    SuperpixelHierarchy SH;
    SH.init(h,w,connect,iterSwitch);
    SH.buildTree(image_shift,edge_shift);


    //Extraction of parameters from the tree
    int * parent = (int *) malloc(h*w*sizeof(int));
    int * label = (int *) malloc(h*w*sizeof(int));
    int * treeu = (int *) malloc((h*w-1)*sizeof(int));
    int * treev = (int *) malloc((h*w-1)*sizeof(int));
    memcpy(parent, SH.getParent(), sizeof(int)*h*w);
    memcpy(label,  SH.getLabel(),  sizeof(int)*h*w);
    memcpy(treeu,  SH.getTreeU(),  sizeof(int)*(h*w-1));
    memcpy(treev,  SH.getTreeV(),  sizeof(int)*(h*w-1));
    int nvertex = h*w;
    int nregion = SH.getRegion();

    printf("Hello\n");
    //Scales you want to get - result store in label
    int superpixels_scale[] = {1000, 500, 250};
    int n = 3;
    char str[100];
    char str2[100];
    for (int i=0; i<n; i++) {

        //Output the label map for a given sp scale
        getSuperpixels(parent,label,treeu,treev,nvertex,nregion,superpixels_scale[i]);

        printf("Hell1qcdso\n");

        //Save output
        char output_map_name[] = "out_test_sp_hier2.png";
        CImg<int> output(w,h);
        for (int i=0; i<w; i++){
            for (int j=0; j<h; j++) {
                output(i,j) = label[j+i*h];
            }
        }

        sprintf(str, "n=%d_", superpixels_scale[i]);
        strcat(str, output_map_name);
        output.save(str);

        char output_border_name[] = "out_border_test_sp_hier.png";
        CImg<> output_border = image;
        int v4x[]={-1,0,1,0};
        int v4y[]={0,-1,0,1};

        cimg_forZ(image,z) {
            cimg_forXY(output_border,x,y) {
                int lab1=output(x,y,z);
                for(int k=0;k<4;k++)
                    if(output_border.containsXYZC(x+v4x[k],y+v4y[k],z))
                        if(lab1 != output(x+v4x[k],y+v4y[k],z))
                            cimg_forC(output_border,c)
                            output_border(x,y,z,c)=0;
            }
        }

        sprintf(str2, "n=%d_", superpixels_scale[i]);
        strcat(str2, output_border_name);
        output_border.save(str2);

    }


    free(parent);
    free(label);
    free(treeu);
    free(treev);
    free(image_shift);
    free(edge_shift);

    return 0;

}
