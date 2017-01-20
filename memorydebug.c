#ifndef _MSC_VER  
#include <sys/time.h> 
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "thnets.h"

#define MAXALLOC 1024*1024*1024

static struct {
	const void *ptr;
	size_t size;
	char file[100];
	int line;
	double t;
} pointers[100000];
static unsigned npointers;
static size_t allocedmem;
static int errordumped;
#ifdef MEMORYDEBUG
static int detailed = MEMORYDEBUG;
#else
static int detailed = 0;
#endif

void debug_memorydump(FILE *fp)
{
	unsigned i;

	fprintf(fp, "%d entries, %lu bytes allocated\n", npointers, (unsigned long)allocedmem);
	for(i = 0; i < npointers; i++)
		fprintf(fp, "%p\t%lu\t%f\t%s:%u\n", pointers[i].ptr, (unsigned long)pointers[i].size, pointers[i].t, pointers[i].file, pointers[i].line);
}

static void debug_print(const char *file, int line, const char *instr, const void *ptr, size_t size, char step)
{
	FILE *fp;

	if(!detailed)
		return;

#ifdef _MSC_VER 
	fopen_s(&fp, "memdump.txt", "a+");
#else  
	fp = fopen("memdump.txt", "a+");
#endif  
	if (fp)
	{
		fprintf(fp, "%f %s:%d\t%s\t%d\t%p\t%lu\n", th_seconds(), file, line, instr, step, ptr, (unsigned long)size);
		fclose(fp);
	}
}

static void debug_addpointer(const void *ptr, size_t size, const char *file, int line)
{
	if(errordumped)
		return;
	pointers[npointers].ptr = ptr;
#ifdef _MSC_VER 
	strcpy_s(pointers[npointers].file, strlen(file), file);
#else  
	strcpy(pointers[npointers].file, file);
#endif  
	 
	pointers[npointers].line = line;
	pointers[npointers].t = th_seconds();
	pointers[npointers++].size = size;
	allocedmem += size;
	if(!errordumped && (allocedmem > MAXALLOC || npointers == 100000))
	{
		errordumped = 1;
		debug_memorydump(stderr);
	}
}

static void debug_removepointer(const void *ptr, const char *file, int line)
{
	unsigned i;

	if(errordumped)
		return;
	for(i = 0; i < npointers; i++)
		if(pointers[i].ptr == ptr)
		{
			allocedmem -= pointers[i].size;
			npointers--;
			memmove(pointers+i, pointers+i+1, (npointers-i)*sizeof(pointers[0]));
			return;
		}
	fprintf(stderr, "Freeing already freed pointer %p in %s:%d\n", ptr, file, line);
}

void *debug_calloc(size_t nmemb, size_t size, const char *file, int line)
{
	void *ptr;

	debug_print(file, line, "calloc", 0, size, 0);
	ptr = calloc(nmemb, size);
	debug_addpointer(ptr, size, file, line);
	debug_print(file, line, "calloc", ptr, size, 1);
	return ptr;
}

void *debug_malloc(size_t size, const char *file, int line)
{
	void *ptr;

	debug_print(file, line, "malloc", 0, size, 0);
	ptr = malloc(size);
	debug_addpointer(ptr, size, file, line);
	debug_print(file, line, "malloc", ptr, size, 1);
	return ptr;
}

void *debug_realloc(void *ptr, size_t size, const char *file, int line)
{
	void *ptr2;

	debug_print(file, line, "realloc", ptr, size, 0);
	debug_removepointer(ptr, file, line);
	ptr2 = realloc(ptr, size);
	debug_addpointer(ptr2, size, file, line);
	debug_print(file, line, "realloc", ptr2, size, 1);
	return ptr2;
}

char *debug_strdup(const char *str, const char *file, int line)
{
	void *ptr;
	int len = strlen(str);

	debug_print(file, line, "strdup", str, len, 0);
#ifdef _MSC_VER 
	ptr = _strdup(str);
#else  
	ptr = strdup(str);
#endif  
	debug_addpointer(ptr, len, file, line);
	debug_print(file, line, "strdup", ptr, len, 1);
	return ptr;
}

void debug_free(void *ptr, const char *file, int line)
{
	debug_print(file, line, "free", ptr, 0, 0);
	free(ptr);
	debug_removepointer(ptr, file, line);
	debug_print(file, line, "free", 0, 0, 1);
}
