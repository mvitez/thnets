#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "thnets.h"

typedef struct {
	char filename[255];
	unsigned char *bitmap;
	int width, height, cp;
} img_t;

double t;

int loadimage(const char *path, img_t *image);

static double seconds()
{
	static double base;
	struct timeval tv;

	gettimeofday(&tv, 0);
	if(!base)
		base = tv.tv_sec + tv.tv_usec * 1e-6;
	return tv.tv_sec + tv.tv_usec * 1e-6 - base;
}

int main(int argc, char **argv)
{
	THNETWORK *net;
	float *result;
	int i, n = 0, rc, outwidth, outheight, runs = 1, print = 0, alg = 2, nbatch = 1;
	const char *modelsdir = 0, *inputfile = 0;

	for(i = 1; i < argc; i++)
	{
		if(argv[i][0] != '-')
			continue;
		switch(argv[i][1])
		{
		case 'm':
			if(i+1 < argc)
				modelsdir = argv[++i];
			break;
		case 'i':
			if(i+1 < argc)
				inputfile = argv[++i];
			break;
		case 'a':
			if(i+1 < argc)
				alg = atoi(argv[++i]);
			break;
		case 'p':
			print = 1;
			break;
		case 'r':
			if(i+1 < argc)
				runs = atoi(argv[++i]);
			break;
		case 'b':
			if(i+1 < argc)
			{
				nbatch = atoi(argv[++i]);
				if(nbatch > 256 || nbatch < 1)
					nbatch = 256;
			}
			break;
		}
	}
	if(!modelsdir || !inputfile)
	{
		fprintf(stderr, "Syntax: test -m <models directory> -i <input file>\n");
		fprintf(stderr, "             [-r <number of runs] [-p(rint results)]\n");
		fprintf(stderr, "             [-a <alg=0:norm,1:MM,2:virtMM (default),3:cuDNN,4:cudNNhalf>]\n");
		fprintf(stderr, "             [-b <nbatch>]\n");
		return -1;
	}
	if(alg == 4)
	{
		alg = 3;
		THCudaHalfFloat(1);
	}
	THInit();
	th_debug = 0;
	net = THLoadNetwork(modelsdir);
	if(net)
	{
		THMakeSpatial(net);
		if(alg == 0)
			THUseSpatialConvolutionMM(net, 0);
		else if(alg == 1 || alg == 2)
			THUseSpatialConvolutionMM(net, alg);
		else if(alg == 3)
		{
			THNETWORK *net2 = THCreateCudaNetwork(net);
			if(!net2)
				THError("CUDA not compiled in");
			THFreeNetwork(net);
			net = net2;
		}
		if(strstr(inputfile, ".t7"))
		{
			struct thobject input_o;

			rc = loadtorch(inputfile, &input_o, 8);
			if(!rc)
			{
				THFloatTensor *in = THFloatTensor_newFromObject(&input_o);
				// In CuDNN the first one has to do some initializations, so don't count it for timing
				if(alg == 3)
					THProcessFloat(net, in->storage->data, 1, in->size[2], in->size[1], &result, &outwidth, &outheight);
				t = seconds();
				for(i = 0; i < runs; i++)
					n = THProcessFloat(net, in->storage->data, 1, in->size[2], in->size[1], &result, &outwidth, &outheight);
				t = (seconds() - t) / runs;
				THFloatTensor_free(in);
				freeobject(&input_o);
			} else printf("Error loading %s\n", inputfile);
		} else {
			img_t image;

			rc = loadimage(inputfile, &image);
			if(!rc)
			{
				unsigned char *bitmaps[256];
				for(i = 0; i < nbatch; i++)
					bitmaps[i] = image.bitmap;
				// In CuDNN the first one has to do some initializations, so don't count it for timing
				if(alg == 3)
					THProcessImages(net, &image.bitmap, 1, image.width, image.height, 3*image.width, &result, &outwidth, &outheight, 0);
				t = seconds();
				for(i = 0; i < runs; i++)
					n = THProcessImages(net, bitmaps, nbatch, image.width, image.height, 3*image.width, &result, &outwidth, &outheight, 0);
				t = (seconds() - t) / runs;
#ifdef USECUDAHOSTALLOC
				cudaFreeHost(image.bitmap);
#else
				free(image.bitmap);
#endif
			} else printf("Error loading image %s\n", inputfile);
		}
		if(print)
			for(i = 0; i < n; i++)
				printf("(%d,%d,%d): %f\n", i/(outwidth*outheight), i % (outwidth*outheight) / outwidth, i % outwidth, result[i]);
		printf("1 run processing time: %lf\n", t);
        THFreeNetwork(net);
	} else printf("The network could not be loaded: %d\n", THLastError());
#ifdef MEMORYDEBUG
	debug_memorydump(stderr);
#endif
	return 0;
}
