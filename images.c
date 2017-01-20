#include <stdlib.h>
#include <stdio.h>
 
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  

#ifdef MEMORYDEBUG
#include "memorydebug.h"
#endif

#ifdef USECUDAHOSTALLOC
#include "thnets.h"



void *cmalloc(size_t size)
{
	void *ptr = 0; 
	errcheck(cudaHostAlloc(&ptr, size, cudaHostAllocMapped));
	return ptr;
}

#else
#define cmalloc(a) malloc(a)
#endif


typedef struct {
	char filename[255];
	unsigned char *bitmap;
	int width, height, cp;
} img_t;
 

static int loadImg(const char *path, img_t *image)
{ 
	int w, h, cp;
	unsigned char *data = stbi_load(path, w, h, cp, 0);
	if ((cp != 3 && cp != 1) || (data == NULL))
	{
		if (data)
		{
			STBI_FREE(data);
		}
		return -1;
	}
	int numberOfPixels = w * h * cp;
	unsigned char *buf = cmalloc(numberOfPixels);
	if (buf == NULL)  	return -1;

	memcpy(buf, data, numberOfPixels);
	STBI_FREE(data); 

	image->bitmap = buf;
	image->width = w;
	image->height = h;
	image->cp = cp;
	return 0;
} 

int loadimage(const char *path, img_t *image)
{
	const char *p = strrchr(path, '.');
	if (!p)
		return -1;
	const char *fn = strrchr(path, '/');
	if (fn)
		fn++;
	else fn = path;
	strcpy(image->filename, fn);
 
	return loadImg(path, image); 
}
