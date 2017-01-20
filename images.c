#include <stdlib.h>
#include <stdio.h>

#ifdef USESTBIMAGE
	#define STB_IMAGE_IMPLEMENTATION
	#include "stb_image.h" 
#else
	#ifndef NOJPEG
	#include <jpeglib.h>
	#endif
	#include <png.h>
#endif

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

#ifdef  USESTBIMAGE

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
#else

#ifndef NOJPEG
static int loadjpeg(const char *path, img_t *image)
{
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	FILE *fp;
	unsigned char *buf;
	int w, h, cp;

	fp = fopen(path, "rb");
	if (!fp)
		return -1;
	cinfo.err = jpeg_std_error(&jerr);
	jpeg_create_decompress(&cinfo);
	jpeg_stdio_src(&cinfo, fp);
	jpeg_read_header(&cinfo, TRUE);
	jpeg_start_decompress(&cinfo);
	w = cinfo.output_width;
	h = cinfo.output_height;
	cp = cinfo.output_components;
	if (cp != 3 && cp != 1)
	{
		jpeg_destroy_decompress(&cinfo);
		fclose(fp);
		return -1;
	}
	buf = cmalloc(w * h * cp);
	while (cinfo.output_scanline < h)
	{
		JSAMPROW buffer = buf + cp*w*cinfo.output_scanline;
		jpeg_read_scanlines(&cinfo, &buffer, 1);
	}
	jpeg_finish_decompress(&cinfo);
	jpeg_destroy_decompress(&cinfo);
	fclose(fp);
	image->bitmap = buf;
	image->width = w;
	image->height = h;
	image->cp = cp;
	return 0;
}
#endif

static int loadpng(const char *path, img_t *image)
{
	png_structp png_ptr;
	png_infop info_ptr;
	FILE *fp;
	unsigned char *buf, header[8];
	int w, h, cp, i, bit_depth;
	png_byte color_type;

	fp = fopen(path, "rb");
	if (!fp)
		return -1;
	if (fread(header, 8, 1, fp) != 1)
	{
		fclose(fp);
		return -1;
	}
	if (png_sig_cmp(header, 0, 8))
	{
		fclose(fp);
		return -1;
	}
	png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	info_ptr = png_create_info_struct(png_ptr);
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		fclose(fp);
		return -1;
	}
	png_init_io(png_ptr, fp);
	png_set_sig_bytes(png_ptr, 8);
	png_read_info(png_ptr, info_ptr);
	w = png_get_image_width(png_ptr, info_ptr);
	h = png_get_image_height(png_ptr, info_ptr);
	color_type = png_get_color_type(png_ptr, info_ptr);
	bit_depth = png_get_bit_depth(png_ptr, info_ptr);
	png_read_update_info(png_ptr, info_ptr);
	switch (color_type)
	{
	case PNG_COLOR_TYPE_RGBA:
		cp = 3;
		png_set_strip_alpha(png_ptr);
		break;
	case PNG_COLOR_TYPE_RGB:
		cp = 3;
		break;
	case PNG_COLOR_TYPE_GRAY:
		if (bit_depth < 8)
			png_set_expand_gray_1_2_4_to_8(png_ptr);
		cp = 1;
		break;
	case PNG_COLOR_TYPE_GA:
		cp = 1;
		png_set_strip_alpha(png_ptr);
		break;
	case PNG_COLOR_TYPE_PALETTE:
		cp = 3;
		png_set_expand(png_ptr);
		break;
	default:
		cp = 0;
	}
	if (cp != 3 && cp != 1)
	{
		fclose(fp);
		return -1;
	}
	if (bit_depth == 16)
		png_set_strip_16(png_ptr);
	buf = cmalloc(w * h * cp);
	png_bytep *rows = (png_bytep *)malloc(sizeof(png_bytep)* h);
	if (setjmp(png_jmpbuf(png_ptr)))
	{
		fclose(fp);
		free(buf);
		free(rows);
		return -1;
	}
	for (i = 0; i < h; i++)
		rows[i] = buf + i * cp * w;
	png_read_image(png_ptr, rows);
	png_read_end(png_ptr, NULL);
	free(rows);
	fclose(fp);
	png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
	image->bitmap = buf;
	image->width = w;
	image->height = h;
	image->cp = cp;
	return 0;
}

#endif
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
#ifdef  USESTBIMAGE
	return loadImg(path, image);
#else
#ifndef NOJPEG
	if (!strcasecmp(p, ".jpeg") || !strcasecmp(p, ".jpg"))
		return loadjpeg(path, image);
#endif
	if (!strcasecmp(p, ".png"))
		return loadpng(path, image);
#endif //  USESTBIMAGE 

	return -1;
}
