#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>

#include "svm.h"
#include "acclibsvm.h"
//int libsvm_version = LIBSVM_VERSION;

#include "parallelthreads.h"
#define NUM_THREADS 8

//#define USE_SSE
//#define USE_AVX
#define ALIGN_DOWN(x, y) (y)*((x)/(y))
#ifdef USE_SSE
#include "xmmintrin.h"
#define SSE_VECTOR_UNROLL 4
#ifdef USE_AVX
#include "immintrin.h"
#define SSE_DIM_ALIGMENT 8
#else
#define SSE_DIM_ALIGMENT 4
#endif
#endif

typedef float Qfloat;
typedef signed char schar;

#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*svm_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*svm_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

//
// AccLibSvmKernel AccLibSvmCache
//
// l is the number of total data items
// size is the cache size limit in bytes
//
class AccLibSvmCache
{
public:
	AccLibSvmCache(int l,size_t size);
	~AccLibSvmCache();

	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int get_data(const int index, Qfloat **data, int len);
	void swap_index(int i, int j);	
private:
	int l;
	size_t size;
	struct head_t
	{
		head_t *prev, *next;	// a circular list
		Qfloat *data;
		int len;		// data[0,len) is cached in this entry
	};

	head_t *head;
	head_t lru_head;
	void lru_delete(head_t *h);
	void lru_insert(head_t *h);
};

AccLibSvmCache::AccLibSvmCache(int l_, size_t size_):l(l_),size(size_)
{
	head = (head_t *)calloc(l,sizeof(head_t));	// initialized to 0
	size /= sizeof(Qfloat);
	size -= l * sizeof(head_t) / sizeof(Qfloat);
	size = max(size, 2 * (size_t) l);	// cache must be large enough for two columns
	lru_head.next = lru_head.prev = &lru_head;
}

AccLibSvmCache::~AccLibSvmCache()
{
	for(head_t *h = lru_head.next; h != &lru_head; h=h->next)
		free(h->data);
	free(head);
}

void AccLibSvmCache::lru_delete(head_t *h)
{
	// delete from current location
	h->prev->next = h->next;
	h->next->prev = h->prev;
}

void AccLibSvmCache::lru_insert(head_t *h)
{
	// insert to last position
	h->next = &lru_head;
	h->prev = lru_head.prev;
	h->prev->next = h;
	h->next->prev = h;
}

int AccLibSvmCache::get_data(const int index, Qfloat **data, int len)
{
	head_t *h = &head[index];
	if(h->len) lru_delete(h);
	int more = len - h->len;

	if(more > 0)
	{
		// free old space
		while(size < more)
		{
			head_t *old = lru_head.next;
			lru_delete(old);
			free(old->data);
			size += old->len;
			old->data = 0;
			old->len = 0;
		}

		// allocate new space
		h->data = (Qfloat *)realloc(h->data,sizeof(Qfloat)*len);
		size -= more;
		swap(h->len,len);
	}

	lru_insert(h);
	*data = h->data;
	return len;
}

void AccLibSvmCache::swap_index(int i, int j)
{
	if(i==j) return;

	if(head[i].len) lru_delete(&head[i]);
	if(head[j].len) lru_delete(&head[j]);
	swap(head[i].data,head[j].data);
	swap(head[i].len,head[j].len);
	if(head[i].len) lru_insert(&head[i]);
	if(head[j].len) lru_insert(&head[j]);

	if(i>j) swap(i,j);
	for(head_t *h = lru_head.next; h!=&lru_head; h=h->next)
	{
		if(h->len > i)
		{
			if(h->len > j)
				swap(h->data[i],h->data[j]);
			else
			{
				// give up
				lru_delete(h);
				free(h->data);
				size += h->len;
				h->data = 0;
				h->len = 0;
			}
		}
	}
}

//
// AccLibSvmKernel evaluation
//
// the static method k_function is for doing single kernel evaluation
// the constructor of AccLibSvmKernel prepares to calculate the l*l kernel matrix
// the member function get_Q is for getting one column from the Q Matrix
//
class AccLibSvmQMatrix {
public:
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const = 0;
	virtual ~AccLibSvmQMatrix() {}
};

class AccLibSvmKernel: public AccLibSvmQMatrix {
public:
	AccLibSvmKernel(int l, acclibsvm_node * x, const libsvm::svm_parameter& param);
	virtual ~AccLibSvmKernel();

	static double k_function(const acclibsvm_node *x, const acclibsvm_node *y,
		const libsvm::svm_parameter& param);
	virtual Qfloat *get_Q(int column, int len) const = 0;
	virtual double *get_QD() const = 0;
	virtual void swap_index(int i, int j) const	// no so const...
	{
		swap(x[i],x[j]);
		if(x_square) swap(x_square[i],x_square[j]);
	}

	void (AccLibSvmKernel::*kernel_function_vect)(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const;
	acclibsvm_node *x;

protected:
	int numThreads;

	double (AccLibSvmKernel::*kernel_function)(int i, int j) const;
	#ifdef USE_SSE
    #ifdef USE_AVX
	static void dot4xAVX(float *result, float *x, float *y0, float *y1, float *y2, float *y3, int dim);
	#endif
	static void dot4xSSE(float *result, float *x, float *y0, float *y1, float *y2, float *y3, int dim);
	#endif
	static void dot4xSparse(float *result, acclibsvm_node *x, acclibsvm_node *y0, acclibsvm_node *y1, acclibsvm_node *y2, acclibsvm_node *y3);


private:

	Qfloat *x_square;

	// svm_parameter
	const int kernel_type;
	const int degree;
	const double gamma;
	const double coef0;

	static double dot(const acclibsvm_node &px, const acclibsvm_node &py);

	double kernel_linear(int i, int j) const
	{
		return dot(x[i],x[j]);
	}
	double kernel_poly(int i, int j) const
	{
		return powi(gamma*dot(x[i],x[j])+coef0,degree);
	}
	double kernel_rbf(int i, int j) const
	{
		return exp(-gamma*(x_square[i]+x_square[j]-2*dot(x[i],x[j])));
	}
	double kernel_sigmoid(int i, int j) const
	{
		return tanh(gamma*dot(x[i],x[j])+coef0);
	}
	double kernel_precomputed(int i, int j) const
	{
		return (x+i)->values[(int)((x+j)->values[0])];
	}

	void kernel_linear_vect_parallel(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const;
	static OUTTYPE kernel_linear_vect_static(INTYPE data);

	void kernel_linear_vect(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const;
	void kernel_poly_vect(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const;
	void kernel_rbf_vect(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const;
	void kernel_sigmoid_vect(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const;
	void kernel_precomputed_vect(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const;

};

#ifdef USE_AVX

void AccLibSvmKernel::dot4xAVX(float *result, float *x, float *y0, float *y1, float *y2, float *y3, int dim) {
    __m256 sum256[SSE_VECTOR_UNROLL];
    __m128 sum128[SSE_VECTOR_UNROLL];
	for(int k=0; k < SSE_VECTOR_UNROLL; k++)
			sum256[k] = _mm256_setzero_ps();

	for(int d=0; d < dim; d+=8) {
		__m256 tmp0,tmp1,tmp2,tmp3;
		__m256 x256 = _mm256_loadu_ps(x+d);
		tmp0 = _mm256_mul_ps(x256, _mm256_loadu_ps(y0+d) );
		tmp1 = _mm256_mul_ps(x256, _mm256_loadu_ps(y1+d) );
		tmp2 = _mm256_mul_ps(x256, _mm256_loadu_ps(y2+d) );
		tmp3 = _mm256_mul_ps(x256, _mm256_loadu_ps(y3+d) );
		sum256[0] = _mm256_add_ps(sum256[0], tmp0);
		sum256[1] = _mm256_add_ps(sum256[1], tmp1);
		sum256[2] = _mm256_add_ps(sum256[2], tmp2);
		sum256[3] = _mm256_add_ps(sum256[3], tmp3);
	}

    sum128[0] = _mm256_extractf128_ps(sum256[0], 0);
    sum128[0] = _mm_add_ps(sum128[0], _mm256_extractf128_ps(sum256[0], 1));
    sum128[1] = _mm256_extractf128_ps(sum256[1], 0);
    sum128[1] = _mm_add_ps(sum128[1], _mm256_extractf128_ps(sum256[1], 1));
    sum128[2] = _mm256_extractf128_ps(sum256[2], 0);
    sum128[2] = _mm_add_ps(sum128[2], _mm256_extractf128_ps(sum256[2], 1));
    sum128[3] = _mm256_extractf128_ps(sum256[3], 0);
    sum128[3] = _mm_add_ps(sum128[3], _mm256_extractf128_ps(sum256[3], 1));

	_MM_TRANSPOSE4_PS(sum128[0], sum128[1], sum128[2], sum128[3]);
	sum128[0] = _mm_add_ps(sum128[0], sum128[1]);
	sum128[2] = _mm_add_ps(sum128[2], sum128[3]);
	sum128[0] = _mm_add_ps(sum128[0], sum128[2]);
	_mm_storeu_ps(result, sum128[0]);
} //dot4xAVX
#endif //USE_AVX

#ifdef USE_SSE
void AccLibSvmKernel::dot4xSSE(float *result, float *x, float *y0, float *y1, float *y2, float *y3, int dim) {
	//SSE_VECTOR_UNROLL 4
	//SSE_DIM_ALIGMENT 4

#if SSE_DIM_ALIGMENT == 4

	__m128 sum128[SSE_VECTOR_UNROLL]; 
	for(int k=0; k < SSE_VECTOR_UNROLL; k++)
			sum128[k] = _mm_setzero_ps();

	for(int d=0; d < dim; d+=4) {
		__m128 tmp0,tmp1,tmp2,tmp3;
		__m128 x128 = _mm_loadu_ps(x+d);
		tmp0 = _mm_mul_ps(x128, _mm_loadu_ps(y0+d) );
		tmp1 = _mm_mul_ps(x128, _mm_loadu_ps(y1+d) );
		tmp2 = _mm_mul_ps(x128, _mm_loadu_ps(y2+d) );
		tmp3 = _mm_mul_ps(x128, _mm_loadu_ps(y3+d) );
		sum128[0] = _mm_add_ps(sum128[0], tmp0);
		sum128[1] = _mm_add_ps(sum128[1], tmp1);
		sum128[2] = _mm_add_ps(sum128[2], tmp2);
		sum128[3] = _mm_add_ps(sum128[3], tmp3);
	}

	_MM_TRANSPOSE4_PS(sum128[0], sum128[1], sum128[2], sum128[3]);
	sum128[0] = _mm_add_ps(sum128[0], sum128[1]);
	sum128[2] = _mm_add_ps(sum128[2], sum128[3]);
	sum128[0] = _mm_add_ps(sum128[0], sum128[2]);
	_mm_storeu_ps(result, sum128[0]);

#else //SSE_DIM_ALIGMENT == 8

	__m128 sumA128[SSE_VECTOR_UNROLL]; 
	__m128 sumB128[SSE_VECTOR_UNROLL]; 
	for(int k=0; k < SSE_VECTOR_UNROLL; k++) {
		sumA128[k] = _mm_setzero_ps();
		sumB128[k] = _mm_setzero_ps();
	}

	for(int d=0; d < dim; d+=8) {
		__m128 tmpA0,tmpA1,tmpA2,tmpA3;
		__m128 tmpB0,tmpB1,tmpB2,tmpB3;
		__m128 xA128 = _mm_loadu_ps(x+d);
		__m128 xB128 = _mm_loadu_ps(x+d+4);
		tmpA0 = _mm_mul_ps(xA128, _mm_loadu_ps(y0+d) );
		tmpA1 = _mm_mul_ps(xA128, _mm_loadu_ps(y1+d) );
		tmpA2 = _mm_mul_ps(xA128, _mm_loadu_ps(y2+d) );
		tmpA3 = _mm_mul_ps(xA128, _mm_loadu_ps(y3+d) );
		tmpB0 = _mm_mul_ps(xB128, _mm_loadu_ps(y0+d+4) );
		tmpB1 = _mm_mul_ps(xB128, _mm_loadu_ps(y1+d+4) );
		tmpB2 = _mm_mul_ps(xB128, _mm_loadu_ps(y2+d+4) );
		tmpB3 = _mm_mul_ps(xB128, _mm_loadu_ps(y3+d+4) );
		sumA128[0] = _mm_add_ps(sumA128[0], tmpA0);
		sumA128[1] = _mm_add_ps(sumA128[1], tmpA1);
		sumA128[2] = _mm_add_ps(sumA128[2], tmpA2);
		sumA128[3] = _mm_add_ps(sumA128[3], tmpA3);
		sumB128[0] = _mm_add_ps(sumB128[0], tmpB0);
		sumB128[1] = _mm_add_ps(sumB128[1], tmpB1);
		sumB128[2] = _mm_add_ps(sumB128[2], tmpB2);
		sumB128[3] = _mm_add_ps(sumB128[3], tmpB3);
	}
	sumA128[0] = _mm_add_ps(sumA128[0], sumB128[0]);
	sumA128[1] = _mm_add_ps(sumA128[1], sumB128[1]);
	sumA128[2] = _mm_add_ps(sumA128[2], sumB128[2]);
	sumA128[3] = _mm_add_ps(sumA128[3], sumB128[3]);
	_MM_TRANSPOSE4_PS(sumA128[0], sumA128[1], sumA128[2], sumA128[3]);
	sumA128[0] = _mm_add_ps(sumA128[0], sumA128[1]);
	sumA128[2] = _mm_add_ps(sumA128[2], sumA128[3]);
	sumA128[0] = _mm_add_ps(sumA128[0], sumA128[2]);
	_mm_storeu_ps(result, sumA128[0]);

#endif //SSE_DIM_ALIGMENT == 4 or 8

} //dot4xSSE
#endif //USE_SSE

void AccLibSvmKernel::dot4xSparse(float *result, acclibsvm_node *x, acclibsvm_node *y0, acclibsvm_node *y1, acclibsvm_node *y2, acclibsvm_node *y3) {
	
	float sum;
	int ix, iy;
	acclibsvm_node *y = NULL;
	for(int v = 0; v < 4; v++) {
		sum = 0;
		ix = 0;
		iy = 0;
		switch(v) {
			case 0: y = y0; break;
			case 1: y = y1; break;
			case 2: y = y2; break;
			case 3: y = y3; break;
		}
		
		if (x->num == 0 || y->num == 0) {
			result[v] = 0;
			continue;
		}

		while(true) {
			if(x->ind[ix] == y->ind[iy]) {
				sum += x->values[ix] == y->values[iy];
				ix++;
				iy++;
			} else {
				if(x->ind[ix] < y->ind[iy]) {
					ix++;
				} else {
					iy++;
				}
			}
			if(ix == x->num) break;
			if(iy == y->num) break;
		}
			
		result[v] = sum;
	}
} //dot4xSparse

void AccLibSvmKernel::kernel_linear_vect(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const {
	kernel_linear_vect_parallel(Qrow, x, ys, start, end);
}

struct data_package {
	Qfloat *Qrow;
	acclibsvm_node *x;
	acclibsvm_node *ys;
	int start;
	int end;
};

void AccLibSvmKernel::kernel_linear_vect_parallel(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const {
	data_package *dps = new data_package[numThreads];
	int numItems = (end-start)/numThreads;
	int _start = start;
	for (int k=0; k < numThreads; k++) {
		dps[k].Qrow = Qrow;
		dps[k].x = x;
		dps[k].ys = ys;
		dps[k].start = _start;
		_start += numItems;
		dps[k].end = _start;
	}
	dps[numThreads-1].end = end;

	if(numItems == 0 || numThreads <= 1) { //too low number of items
		dps[0].end = end;
		kernel_linear_vect_static((void *)dps);
		delete [] dps;
		return;
	}

	ParallelThreads::run_parallel(AccLibSvmKernel::kernel_linear_vect_static, dps, sizeof(data_package), numThreads);

	delete [] dps;
}

OUTTYPE AccLibSvmKernel::kernel_linear_vect_static(INTYPE _data_package) {
	data_package *dp = (data_package*) _data_package;
	Qfloat *Qrow = dp->Qrow;
	acclibsvm_node *x = dp->x;
	acclibsvm_node *ys = dp->ys;
	int start = dp->start;
	int end = dp->end;

#if defined(USE_SSE) || defined(USE_AVX)
	//SSE_VECTOR_UNROLL 4
	const int dim = x->num;
	int j = start;
	for(; j < start+ALIGN_DOWN(end-start, SSE_VECTOR_UNROLL); j+=SSE_VECTOR_UNROLL) { //SSE_VECTOR_UNROLL vectors at once
		//Qrow[j] = (Qfloat) dot(x[i], x[j]);
		//Qrow[j+1] = (Qfloat) dot(x[i], x[j+1]);
		//Qrow[j+2] = (Qfloat) dot(x[i], x[j+2]);
		//Qrow[j+3] = (Qfloat) dot(x[i], x[j+3]);
	if(x->ind == NULL) //dense
#ifdef USE_AVX
        dot4xAVX(Qrow+j, x->values, ys[j].values, ys[j+1].values, ys[j+2].values, ys[j+3].values, dim);
#else
		dot4xSSE(Qrow+j, x->values, ys[j].values, ys[j+1].values, ys[j+2].values, ys[j+3].values, dim);
#endif
	else
		dot4xSparse(Qrow+j, x, &ys[j], &ys[j+1], &ys[j+2], &ys[j+3]);
	}
	if(j < end) { //the unaligned rest
		float tmp[SSE_VECTOR_UNROLL];
		if(x->ind == NULL) { //dense
#ifdef USE_AVX
        dot4xAVX(tmp, x->values, ys[j].values, ys[min(j, end-1)].values, ys[min(j, end-1)].values, ys[min(j, end-1)].values, dim);
#else
		dot4xSSE(tmp, x->values, ys[j].values, ys[min(j+1, end-1)].values, ys[min(j+2, end-1)].values, ys[min(j+3, end-1)].values, dim);
#endif
		} else
			dot4xSparse(tmp, x, &ys[j], &ys[min(j+1, end-1)], &ys[min(j+2, end-1)], &ys[min(j+3, end-1)]);
		//for(int k = j; k < end; k++) Qrow[k] = (Qfloat) dot(*x, ys[k]);
		memcpy(Qrow+j, tmp, (end-j)*sizeof(float)); //copy the valid part
	}
#else
	for(int j=start; j < end; j++) Qrow[j] = (Qfloat)dot(*x, ys[j]);
#endif
	return NULL;
} //kernel_linear_vect_static

void AccLibSvmKernel::kernel_poly_vect(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const {
	kernel_linear_vect_parallel(Qrow, x, ys, start, end);
	for(int j=start; j < end; j++) Qrow[j] = (Qfloat)powi((Qfloat)gamma*Qrow[j]+(Qfloat)coef0,degree);
} //kernel_poly_vect

void AccLibSvmKernel::kernel_rbf_vect(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const {
	kernel_linear_vect_parallel(Qrow, x, ys, start, end);
	unsigned int i = (unsigned int)(x-ys); //we don't have i, need to calculate
	for(int j=start; j < end; j++) Qrow[j] = expf(-(Qfloat)gamma*(x_square[i]+x_square[j]-2*Qrow[j]));
} //kernel_rbf_vect

void AccLibSvmKernel::kernel_sigmoid_vect(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const {
	kernel_linear_vect_parallel(Qrow, x, ys, start, end);
	for(int j=start; j < end; j++) Qrow[j] = tanh((Qfloat)gamma*Qrow[j]+(Qfloat)coef0);
} //kernel_sigmoid_vect

void AccLibSvmKernel::kernel_precomputed_vect(Qfloat *Qrow, acclibsvm_node *x, acclibsvm_node *ys, int start, int end) const {
	for(int j=start; j < end; j++) Qrow[j] = (Qfloat)x->values[(int)ys[j].values[0]];
} //kernel_precomputed_vect


using namespace libsvm;

AccLibSvmKernel::AccLibSvmKernel(int l, acclibsvm_node * x_, const libsvm::svm_parameter& param)
:kernel_type(param.kernel_type), degree(param.degree),
 gamma(param.gamma), coef0(param.coef0)
{
	switch(kernel_type)
	{
		case LINEAR:
			kernel_function = &AccLibSvmKernel::kernel_linear;
			break;
		case POLY:
			kernel_function = &AccLibSvmKernel::kernel_poly;
			break;
		case RBF:
			kernel_function = &AccLibSvmKernel::kernel_rbf;
			break;
		case SIGMOID:
			kernel_function = &AccLibSvmKernel::kernel_sigmoid;
			break;
		case PRECOMPUTED:
			kernel_function = &AccLibSvmKernel::kernel_precomputed;
			break;
	}

	switch(kernel_type)
	{
		case LINEAR:
			kernel_function_vect = &AccLibSvmKernel::kernel_linear_vect;
			break;
		case POLY:
			kernel_function_vect = &AccLibSvmKernel::kernel_poly_vect;
			break;
		case RBF:
			kernel_function_vect = &AccLibSvmKernel::kernel_rbf_vect;
			break;
		case SIGMOID:
			kernel_function_vect = &AccLibSvmKernel::kernel_sigmoid_vect;
			break;
		case PRECOMPUTED:
			kernel_function_vect = &AccLibSvmKernel::kernel_precomputed_vect;
			break;
	}

	clone(x,x_,l);

	if(kernel_type == RBF)
	{
		x_square = new Qfloat[l];
		for(int i=0;i<l;i++)
			x_square[i] = (Qfloat)dot(x[i],x[i]);
	}
	else
		x_square = 0;

#ifdef USE_PTHREADS
	numThreads = NUM_THREADS;
#else
    numThreads = std::thread::hardware_concurrency();
#endif
}

AccLibSvmKernel::~AccLibSvmKernel()
{
	delete[] x;
	delete[] x_square;
}


double AccLibSvmKernel::dot(const acclibsvm_node &px, const acclibsvm_node &py)
{
	double sum = 0;

	if(px.ind == NULL) { //DENSE
		int dim = min(px.num, py.num);
		for (int i = 0; i < dim; i++)
			sum += px.values[i] * py.values[i];
	} else { //SPARSE
		int ix=0, iy=0;
		for(; ix < px.num && iy < py.num;)
		{
			if(px.ind[ix] == py.ind[iy])
			{
				sum += px.values[ix] * py.values[iy];
				++ix;
				++iy;
			}
			else
			{
				if(px.ind[ix] > py.ind[iy])
				{	
					++iy;
				}
				else
				{
					++ix;
				}
			}
		}
	}
	return sum;
}

/*
double AccLibSvmKernel::dot(const acclibsvm_node *px, const acclibsvm_node *py)
{
	double sum = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += px->value * py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
	return sum;
}
*/


double AccLibSvmKernel::k_function(const acclibsvm_node *x, const acclibsvm_node *y,
			  const svm_parameter& param)
{
	switch(param.kernel_type)
	{
		case LINEAR:
			return dot(*x,*y);
		case POLY:
			return powi(param.gamma*dot(*x,*y)+param.coef0,param.degree);
		case RBF:
		{
			double sum = 0;
			if(x->ind == NULL) { //DENSE
				int dim = min(x->num, y->num), i;
				for (i = 0; i < dim; i++)
				{
					double d = x->values[i] - y->values[i];
					sum += d*d;
				}
				for (; i < x->num; i++)
					sum += x->values[i] * x->values[i];
				for (; i < y->num; i++)
					sum += y->values[i] * y->values[i];
			} else { //SPARSE

				int ix=0, iy=0;
				for(; ix < x->num && iy < y->num;)
				{
					if(x->ind[ix] == y->ind[iy])
					{
						double d = x->values[ix] - y->values[iy];
						sum += d*d;
						++ix;
						++iy;
					}
					else
					{
						if(x->ind[ix] > y->ind[iy])
						{	
							sum += y->values[iy] * y->values[iy];
							++iy;
						}
						else
						{
							sum += x->values[ix] * x->values[ix];
							++ix;
						}
					}
				}

				for (; ix < x->num; ix++) sum += x->values[ix] * x->values[ix];
				for (; iy < y->num; iy++) sum += y->values[iy] * y->values[iy];
			}
			return exp(-param.gamma*sum);
		}
		case SIGMOID:
			return tanh(param.gamma*dot(*x,*y)+param.coef0);
		case PRECOMPUTED:  //x: test (validation), y: SV
			return x->values[(int)(y->values[0])];
		default:
			return 0;  // Unreachable 
	}
}

// An SMO algorithm in Fan et al., JMLR 6(2005), p. 1889--1918
// Solves:
//
//	min 0.5(\alpha^T Q \alpha) + p^T \alpha
//
//		y^T \alpha = \delta
//		y_i = +1 or -1
//		0 <= alpha_i <= Cp for y_i = 1
//		0 <= alpha_i <= Cn for y_i = -1
//
// Given:
//
//	Q, p, y, Cp, Cn, and an initial feasible point \alpha
//	l is the size of vectors and matrices
//	eps is the stopping tolerance
//
// solution will be put in \alpha, objective value will be put in obj
//
class AccLibSvmSolver {
public:
	AccLibSvmSolver() {};
	virtual ~AccLibSvmSolver() {};

	struct SolutionInfo {
		double obj;
		double rho;
		double upper_bound_p;
		double upper_bound_n;
		double r;	// for AccLibSvmSolver_NU
	};

	void Solve(int l, const AccLibSvmQMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking);
protected:
	int active_size;
	schar *y;
	double *G;		// gradient of objective function
	enum { LOWER_BOUND, UPPER_BOUND, FREE };
	char *alpha_status;	// LOWER_BOUND, UPPER_BOUND, FREE
	double *alpha;
	const AccLibSvmQMatrix *Q;
	const double *QD;
	double eps;
	double Cp,Cn;
	double *p;
	int *active_set;
	double *G_bar;		// gradient, if we treat free variables as 0
	int l;
	bool unshrink;	// XXX

	double get_C(int i)
	{
		return (y[i] > 0)? Cp : Cn;
	}
	void update_alpha_status(int i)
	{
		if(alpha[i] >= get_C(i))
			alpha_status[i] = UPPER_BOUND;
		else if(alpha[i] <= 0)
			alpha_status[i] = LOWER_BOUND;
		else alpha_status[i] = FREE;
	}
	bool is_upper_bound(int i) { return alpha_status[i] == UPPER_BOUND; }
	bool is_lower_bound(int i) { return alpha_status[i] == LOWER_BOUND; }
	bool is_free(int i) { return alpha_status[i] == FREE; }
	void swap_index(int i, int j);
	void reconstruct_gradient();
	virtual int select_working_set(int &i, int &j);
	virtual double calculate_rho();
	virtual void do_shrinking();
private:
	bool be_shrunk(int i, double Gmax1, double Gmax2);	
};

void AccLibSvmSolver::swap_index(int i, int j)
{
	Q->swap_index(i,j);
	swap(y[i],y[j]);
	swap(G[i],G[j]);
	swap(alpha_status[i],alpha_status[j]);
	swap(alpha[i],alpha[j]);
	swap(p[i],p[j]);
	swap(active_set[i],active_set[j]);
	swap(G_bar[i],G_bar[j]);
}

void AccLibSvmSolver::reconstruct_gradient()
{
	// reconstruct inactive elements of G from G_bar and free variables

	if(active_size == l) return;

	int i,j;
	int nr_free = 0;

	for(j=active_size;j<l;j++)
		G[j] = G_bar[j] + p[j];

	for(j=0;j<active_size;j++)
		if(is_free(j))
			nr_free++;

	if(2*nr_free < active_size)
		info("\nWARNING: using -h 0 may be faster\n");

	if (nr_free*l > 2*active_size*(l-active_size))
	{
		for(i=active_size;i<l;i++)
		{
			const Qfloat *Q_i = Q->get_Q(i,active_size);
			for(j=0;j<active_size;j++)
				if(is_free(j))
					G[i] += alpha[j] * Q_i[j];
		}
	}
	else
	{
		for(i=0;i<active_size;i++)
			if(is_free(i))
			{
				const Qfloat *Q_i = Q->get_Q(i,l);
				double alpha_i = alpha[i];
				for(j=active_size;j<l;j++)
					G[j] += alpha_i * Q_i[j];
			}
	}
}

void AccLibSvmSolver::Solve(int l, const AccLibSvmQMatrix& Q, const double *p_, const schar *y_,
		   double *alpha_, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
{
	this->l = l;
	this->Q = &Q;
	QD=Q.get_QD();
	clone(p, p_,l);
	clone(y, y_,l);
	clone(alpha,alpha_,l);
	this->Cp = Cp;
	this->Cn = Cn;
	this->eps = eps;
	unshrink = false;

	// initialize alpha_status
	{
		alpha_status = new char[l];
		for(int i=0;i<l;i++)
			update_alpha_status(i);
	}

	// initialize active set (for shrinking)
	{
		active_set = new int[l];
		for(int i=0;i<l;i++)
			active_set[i] = i;
		active_size = l;
	}

	// initialize gradient
	{
		G = new double[l];
		G_bar = new double[l];
		int i;
		for(i=0;i<l;i++)
		{
			G[i] = p[i];
			G_bar[i] = 0;
		}
		for(i=0;i<l;i++)
			if(!is_lower_bound(i))
			{
				const Qfloat *Q_i = Q.get_Q(i,l);
				double alpha_i = alpha[i];
				int j;
				for(j=0;j<l;j++)
					G[j] += alpha_i*Q_i[j];
				if(is_upper_bound(i))
					for(j=0;j<l;j++)
						G_bar[j] += get_C(i) * Q_i[j];
			}
	}

	// optimization step
	
	int iter = 0;
	int max_iter = max(10000000, l>INT_MAX/100 ? INT_MAX : 100*l);
	int counter = min(l,1000)+1;

	while(iter < max_iter)
	{
		// show progress and do shrinking

		if(--counter == 0)
		{
			counter = min(l,1000);
			if(shrinking) do_shrinking();
			info(".");
		}

		int i,j;
		if(select_working_set(i,j)!=0)
		{
			// reconstruct the whole gradient
			reconstruct_gradient();
			// reset active set size and check
			active_size = l;
			info("*");
			if(select_working_set(i,j)!=0)
				break;
			else
				counter = 1;	// do shrinking next iteration
		}
		
		++iter;
		
		// update alpha[i] and alpha[j], handle bounds carefully
		
		const Qfloat *Q_i = Q.get_Q(i,active_size);
		const Qfloat *Q_j = Q.get_Q(j,active_size);

		double C_i = get_C(i);
		double C_j = get_C(j);

		double old_alpha_i = alpha[i];
		double old_alpha_j = alpha[j];

		if(y[i]!=y[j])
		{
			double quad_coef = QD[i]+QD[j]+2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (-G[i]-G[j])/quad_coef;
			double diff = alpha[i] - alpha[j];
			alpha[i] += delta;
			alpha[j] += delta;
			
			if(diff > 0)
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = diff;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = -diff;
				}
			}
			if(diff > C_i - C_j)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = C_i - diff;
				}
			}
			else
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = C_j + diff;
				}
			}
		}
		else
		{
			double quad_coef = QD[i]+QD[j]-2*Q_i[j];
			if (quad_coef <= 0)
				quad_coef = TAU;
			double delta = (G[i]-G[j])/quad_coef;
			double sum = alpha[i] + alpha[j];
			alpha[i] -= delta;
			alpha[j] += delta;

			if(sum > C_i)
			{
				if(alpha[i] > C_i)
				{
					alpha[i] = C_i;
					alpha[j] = sum - C_i;
				}
			}
			else
			{
				if(alpha[j] < 0)
				{
					alpha[j] = 0;
					alpha[i] = sum;
				}
			}
			if(sum > C_j)
			{
				if(alpha[j] > C_j)
				{
					alpha[j] = C_j;
					alpha[i] = sum - C_j;
				}
			}
			else
			{
				if(alpha[i] < 0)
				{
					alpha[i] = 0;
					alpha[j] = sum;
				}
			}
		}

		// update G

		double delta_alpha_i = alpha[i] - old_alpha_i;
		double delta_alpha_j = alpha[j] - old_alpha_j;
		
		for(int k=0;k<active_size;k++)
		{
			G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
		}

		// update alpha_status and G_bar

		{
			bool ui = is_upper_bound(i);
			bool uj = is_upper_bound(j);
			update_alpha_status(i);
			update_alpha_status(j);
			int k;
			if(ui != is_upper_bound(i))
			{
				Q_i = Q.get_Q(i,l);
				if(ui)
					for(k=0;k<l;k++)
						G_bar[k] -= C_i * Q_i[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_i * Q_i[k];
			}

			if(uj != is_upper_bound(j))
			{
				Q_j = Q.get_Q(j,l);
				if(uj)
					for(k=0;k<l;k++)
						G_bar[k] -= C_j * Q_j[k];
				else
					for(k=0;k<l;k++)
						G_bar[k] += C_j * Q_j[k];
			}
		}
	}

	if(iter >= max_iter)
	{
		if(active_size < l)
		{
			// reconstruct the whole gradient to calculate objective value
			reconstruct_gradient();
			active_size = l;
			info("*");
		}
		fprintf(stderr,"\nWARNING: reaching max number of iterations\n");
	}
	
	// calculate rho

	si->rho = calculate_rho();

	// calculate objective value
	{
		double v = 0;
		int i;
		for(i=0;i<l;i++)
			v += alpha[i] * (G[i] + p[i]);

		si->obj = v/2;
	}

	// put back the solution
	{
		for(int i=0;i<l;i++)
			alpha_[active_set[i]] = alpha[i];
	}

	// juggle everything back
	/*{
		for(int i=0;i<l;i++)
			while(active_set[i] != i)
				swap_index(i,active_set[i]);
				// or Q.swap_index(i,active_set[i]);
	}*/

	si->upper_bound_p = Cp;
	si->upper_bound_n = Cn;

	info("\noptimization finished, #iter = %d\n",iter);

	delete[] p;
	delete[] y;
	delete[] alpha;
	delete[] alpha_status;
	delete[] active_set;
	delete[] G;
	delete[] G_bar;
    ParallelThreads::terminate_threads();
}

// return 1 if already optimal, return 0 otherwise
int AccLibSvmSolver::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
	
	double Gmax = -INF;
	double Gmax2 = -INF;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)	
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmax)
				{
					Gmax = -G[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmax)
				{
					Gmax = G[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const Qfloat *Q_i = NULL;
	if(i != -1) // NULL Q_i not accessed: Gmax=-INF if i=-1
		Q_i = Q->get_Q(i,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))
			{
				double grad_diff=Gmax+G[j];
				if (G[j] >= Gmax2)
					Gmax2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff= Gmax-G[j];
				if (-G[j] >= Gmax2)
					Gmax2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(Gmax+Gmax2 < eps)
		return 1;

	out_i = Gmax_idx;
	out_j = Gmin_idx;
	return 0;
}

bool AccLibSvmSolver::be_shrunk(int i, double Gmax1, double Gmax2)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else
			return(-G[i] > Gmax2);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax1);
	}
	else
		return(false);
}

void AccLibSvmSolver::do_shrinking()
{
	int i;
	double Gmax1 = -INF;		// max { -y_i * grad(f)_i | i in I_up(\alpha) }
	double Gmax2 = -INF;		// max { y_i * grad(f)_i | i in I_low(\alpha) }

	// find maximal violating pair first
	for(i=0;i<active_size;i++)
	{
		if(y[i]==+1)	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax1)
					Gmax1 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax2)
					Gmax2 = G[i];
			}
		}
		else	
		{
			if(!is_upper_bound(i))	
			{
				if(-G[i] >= Gmax2)
					Gmax2 = -G[i];
			}
			if(!is_lower_bound(i))	
			{
				if(G[i] >= Gmax1)
					Gmax1 = G[i];
			}
		}
	}

	if(unshrink == false && Gmax1 + Gmax2 <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
		info("*");
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double AccLibSvmSolver::calculate_rho()
{
	double r;
	int nr_free = 0;
	double ub = INF, lb = -INF, sum_free = 0;
	for(int i=0;i<active_size;i++)
	{
		double yG = y[i]*G[i];

		if(is_upper_bound(i))
		{
			if(y[i]==-1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else if(is_lower_bound(i))
		{
			if(y[i]==+1)
				ub = min(ub,yG);
			else
				lb = max(lb,yG);
		}
		else
		{
			++nr_free;
			sum_free += yG;
		}
	}

	if(nr_free>0)
		r = sum_free/nr_free;
	else
		r = (ub+lb)/2;

	return r;
}

//
// AccLibSvmSolver for nu-svm classification and regression
//
// additional constraint: e^T \alpha = constant
//
class AccLibSvmSolver_NU: public AccLibSvmSolver
{
public:
	AccLibSvmSolver_NU() {}
	void Solve(int l, const AccLibSvmQMatrix& Q, const double *p, const schar *y,
		   double *alpha, double Cp, double Cn, double eps,
		   SolutionInfo* si, int shrinking)
	{
		this->si = si;
		AccLibSvmSolver::Solve(l,Q,p,y,alpha,Cp,Cn,eps,si,shrinking);
	}
private:
	SolutionInfo *si;
	int select_working_set(int &i, int &j);
	double calculate_rho();
	bool be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4);
	void do_shrinking();
};

// return 1 if already optimal, return 0 otherwise
int AccLibSvmSolver_NU::select_working_set(int &out_i, int &out_j)
{
	// return i,j such that y_i = y_j and
	// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
	// j: minimizes the decrease of obj value
	//    (if quadratic coefficeint <= 0, replace it with tau)
	//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)

	double Gmaxp = -INF;
	double Gmaxp2 = -INF;
	int Gmaxp_idx = -1;

	double Gmaxn = -INF;
	double Gmaxn2 = -INF;
	int Gmaxn_idx = -1;

	int Gmin_idx = -1;
	double obj_diff_min = INF;

	for(int t=0;t<active_size;t++)
		if(y[t]==+1)
		{
			if(!is_upper_bound(t))
				if(-G[t] >= Gmaxp)
				{
					Gmaxp = -G[t];
					Gmaxp_idx = t;
				}
		}
		else
		{
			if(!is_lower_bound(t))
				if(G[t] >= Gmaxn)
				{
					Gmaxn = G[t];
					Gmaxn_idx = t;
				}
		}

	int ip = Gmaxp_idx;
	int in = Gmaxn_idx;
	const Qfloat *Q_ip = NULL;
	const Qfloat *Q_in = NULL;
	if(ip != -1) // NULL Q_ip not accessed: Gmaxp=-INF if ip=-1
		Q_ip = Q->get_Q(ip,active_size);
	if(in != -1)
		Q_in = Q->get_Q(in,active_size);

	for(int j=0;j<active_size;j++)
	{
		if(y[j]==+1)
		{
			if (!is_lower_bound(j))	
			{
				double grad_diff=Gmaxp+G[j];
				if (G[j] >= Gmaxp2)
					Gmaxp2 = G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef = QD[ip]+QD[j]-2*Q_ip[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (!is_upper_bound(j))
			{
				double grad_diff=Gmaxn-G[j];
				if (-G[j] >= Gmaxn2)
					Gmaxn2 = -G[j];
				if (grad_diff > 0)
				{
					double obj_diff; 
					double quad_coef = QD[in]+QD[j]-2*Q_in[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/TAU;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(max(Gmaxp+Gmaxp2,Gmaxn+Gmaxn2) < eps)
		return 1;

	if (y[Gmin_idx] == +1)
		out_i = Gmaxp_idx;
	else
		out_i = Gmaxn_idx;
	out_j = Gmin_idx;

	return 0;
}

bool AccLibSvmSolver_NU::be_shrunk(int i, double Gmax1, double Gmax2, double Gmax3, double Gmax4)
{
	if(is_upper_bound(i))
	{
		if(y[i]==+1)
			return(-G[i] > Gmax1);
		else	
			return(-G[i] > Gmax4);
	}
	else if(is_lower_bound(i))
	{
		if(y[i]==+1)
			return(G[i] > Gmax2);
		else	
			return(G[i] > Gmax3);
	}
	else
		return(false);
}

void AccLibSvmSolver_NU::do_shrinking()
{
	double Gmax1 = -INF;	// max { -y_i * grad(f)_i | y_i = +1, i in I_up(\alpha) }
	double Gmax2 = -INF;	// max { y_i * grad(f)_i | y_i = +1, i in I_low(\alpha) }
	double Gmax3 = -INF;	// max { -y_i * grad(f)_i | y_i = -1, i in I_up(\alpha) }
	double Gmax4 = -INF;	// max { y_i * grad(f)_i | y_i = -1, i in I_low(\alpha) }

	// find maximal violating pair first
	int i;
	for(i=0;i<active_size;i++)
	{
		if(!is_upper_bound(i))
		{
			if(y[i]==+1)
			{
				if(-G[i] > Gmax1) Gmax1 = -G[i];
			}
			else	if(-G[i] > Gmax4) Gmax4 = -G[i];
		}
		if(!is_lower_bound(i))
		{
			if(y[i]==+1)
			{	
				if(G[i] > Gmax2) Gmax2 = G[i];
			}
			else	if(G[i] > Gmax3) Gmax3 = G[i];
		}
	}

	if(unshrink == false && max(Gmax1+Gmax2,Gmax3+Gmax4) <= eps*10) 
	{
		unshrink = true;
		reconstruct_gradient();
		active_size = l;
	}

	for(i=0;i<active_size;i++)
		if (be_shrunk(i, Gmax1, Gmax2, Gmax3, Gmax4))
		{
			active_size--;
			while (active_size > i)
			{
				if (!be_shrunk(active_size, Gmax1, Gmax2, Gmax3, Gmax4))
				{
					swap_index(i,active_size);
					break;
				}
				active_size--;
			}
		}
}

double AccLibSvmSolver_NU::calculate_rho()
{
	int nr_free1 = 0,nr_free2 = 0;
	double ub1 = INF, ub2 = INF;
	double lb1 = -INF, lb2 = -INF;
	double sum_free1 = 0, sum_free2 = 0;

	for(int i=0;i<active_size;i++)
	{
		if(y[i]==+1)
		{
			if(is_upper_bound(i))
				lb1 = max(lb1,G[i]);
			else if(is_lower_bound(i))
				ub1 = min(ub1,G[i]);
			else
			{
				++nr_free1;
				sum_free1 += G[i];
			}
		}
		else
		{
			if(is_upper_bound(i))
				lb2 = max(lb2,G[i]);
			else if(is_lower_bound(i))
				ub2 = min(ub2,G[i]);
			else
			{
				++nr_free2;
				sum_free2 += G[i];
			}
		}
	}

	double r1,r2;
	if(nr_free1 > 0)
		r1 = sum_free1/nr_free1;
	else
		r1 = (ub1+lb1)/2;
	
	if(nr_free2 > 0)
		r2 = sum_free2/nr_free2;
	else
		r2 = (ub2+lb2)/2;
	
	si->r = (r1+r2)/2;
	return (r1-r2)/2;
}

//
// Q matrices for various formulations
//
class AccLibSvmSVC_Q: public AccLibSvmKernel
{ 
public:
	AccLibSvmSVC_Q(const acclibsvm_problem& prob, const svm_parameter& param, const schar *y_)
	:AccLibSvmKernel(prob.l, prob.x, param)
	{
		clone(y,y_,prob.l);
		cache = new AccLibSvmCache(prob.l,(size_t)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start, j;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			(this->*kernel_function_vect)(data, &x[i], x, start, len);
			for(j=start;j<len;j++) data[j] *= y[i] * y[j];
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		AccLibSvmKernel::swap_index(i,j);
		swap(y[i],y[j]);
		swap(QD[i],QD[j]);
	}

	~AccLibSvmSVC_Q()
	{
		delete[] y;
		delete cache;
		delete[] QD;
	}
private:
	schar *y;
	AccLibSvmCache *cache;
	double *QD;
};

class ONE_CLASS_Q: public AccLibSvmKernel
{
public:
	ONE_CLASS_Q(const acclibsvm_problem& prob, const svm_parameter& param)
	:AccLibSvmKernel(prob.l, prob.x, param)
	{
		cache = new AccLibSvmCache(prob.l,(size_t)(param.cache_size*(1<<20)));
		QD = new double[prob.l];
		for(int i=0;i<prob.l;i++)
			QD[i] = (this->*kernel_function)(i,i);
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int start;
		if((start = cache->get_data(i,&data,len)) < len)
		{
			(this->*kernel_function_vect)(data, &x[i], x, start, len);
		}
		return data;
	}

	double *get_QD() const
	{
		return QD;
	}

	void swap_index(int i, int j) const
	{
		cache->swap_index(i,j);
		AccLibSvmKernel::swap_index(i,j);
		swap(QD[i],QD[j]);
	}

	~ONE_CLASS_Q()
	{
		delete cache;
		delete[] QD;
	}
private:
	AccLibSvmCache *cache;
	double *QD;
};

class SVR_Q: public AccLibSvmKernel
{ 
public:
	SVR_Q(const acclibsvm_problem& prob, const svm_parameter& param)
	:AccLibSvmKernel(prob.l, prob.x, param)
	{
		l = prob.l;
		cache = new AccLibSvmCache(l,(size_t)(param.cache_size*(1<<20)));
		QD = new double[2*l];
		sign = new schar[2*l];
		index = new int[2*l];
		for(int k=0;k<l;k++)
		{
			sign[k] = 1;
			sign[k+l] = -1;
			index[k] = k;
			index[k+l] = k;
			QD[k] = (this->*kernel_function)(k,k);
			QD[k+l] = QD[k];
		}
		buffer[0] = new Qfloat[2*l];
		buffer[1] = new Qfloat[2*l];
		next_buffer = 0;
	}

	void swap_index(int i, int j) const
	{
		swap(sign[i],sign[j]);
		swap(index[i],index[j]);
		swap(QD[i],QD[j]);
	}
	
	Qfloat *get_Q(int i, int len) const
	{
		Qfloat *data;
		int j, real_i = index[i];
		if(cache->get_data(real_i,&data,l) < l)
		{
			(this->*kernel_function_vect)(data, &x[real_i], x, 0, l);
		}

		// reorder and copy
		Qfloat *buf = buffer[next_buffer];
		next_buffer = 1 - next_buffer;
		schar si = sign[i];
		for(j=0;j<len;j++)
			buf[j] = (Qfloat) si * (Qfloat) sign[j] * data[index[j]];
		return buf;
	}

	double *get_QD() const
	{
		return QD;
	}

	~SVR_Q()
	{
		delete cache;
		delete[] sign;
		delete[] index;
		delete[] buffer[0];
		delete[] buffer[1];
		delete[] QD;
	}
private:
	int l;
	AccLibSvmCache *cache;
	schar *sign;
	int *index;
	mutable int next_buffer;
	Qfloat *buffer[2];
	double *QD;
};

//
// construct and solve various formulations
//
static void acclibsvm_solve_c_svc(
	const acclibsvm_problem *prob, const svm_parameter* param,
	double *alpha, AccLibSvmSolver::SolutionInfo* si, double Cp, double Cn)
{
	int l = prob->l;
	double *minus_ones = new double[l];
	schar *y = new schar[l];

	int i;

	for(i=0;i<l;i++)
	{
		alpha[i] = 0;
		minus_ones[i] = -1;
		if(prob->y[i] > 0) y[i] = +1; else y[i] = -1;
	}

	AccLibSvmSolver s;
	s.Solve(l, AccLibSvmSVC_Q(*prob,*param,y), minus_ones, y,
		alpha, Cp, Cn, param->eps, si, param->shrinking);

	double sum_alpha=0;
	for(i=0;i<l;i++)
		sum_alpha += alpha[i];

	if (Cp==Cn)
		info("nu = %f\n", sum_alpha/(Cp*prob->l));

	for(i=0;i<l;i++)
		alpha[i] *= y[i];

	delete[] minus_ones;
	delete[] y;
}

static void acclibsvm_solve_nu_svc(
	const acclibsvm_problem *prob, const svm_parameter *param,
	double *alpha, AccLibSvmSolver::SolutionInfo* si)
{
	int i;
	int l = prob->l;
	double nu = param->nu;

	schar *y = new schar[l];

	for(i=0;i<l;i++)
		if(prob->y[i]>0)
			y[i] = +1;
		else
			y[i] = -1;

	double sum_pos = nu*l/2;
	double sum_neg = nu*l/2;

	for(i=0;i<l;i++)
		if(y[i] == +1)
		{
			alpha[i] = min(1.0,sum_pos);
			sum_pos -= alpha[i];
		}
		else
		{
			alpha[i] = min(1.0,sum_neg);
			sum_neg -= alpha[i];
		}

	double *zeros = new double[l];

	for(i=0;i<l;i++)
		zeros[i] = 0;

	AccLibSvmSolver_NU s;
	s.Solve(l, AccLibSvmSVC_Q(*prob,*param,y), zeros, y,
		alpha, 1.0, 1.0, param->eps, si,  param->shrinking);
	double r = si->r;

	info("C = %f\n",1/r);

	for(i=0;i<l;i++)
		alpha[i] *= y[i]/r;

	si->rho /= r;
	si->obj /= (r*r);
	si->upper_bound_p = 1/r;
	si->upper_bound_n = 1/r;

	delete[] y;
	delete[] zeros;
}

static void acclibsvm_solve_one_class(
	const acclibsvm_problem *prob, const svm_parameter *param,
	double *alpha, AccLibSvmSolver::SolutionInfo* si)
{
	int l = prob->l;
	double *zeros = new double[l];
	schar *ones = new schar[l];
	int i;

	int n = (int)(param->nu*prob->l);	// # of alpha's at upper bound

	for(i=0;i<n;i++)
		alpha[i] = 1;
	if(n<prob->l)
		alpha[n] = param->nu * prob->l - n;
	for(i=n+1;i<l;i++)
		alpha[i] = 0;

	for(i=0;i<l;i++)
	{
		zeros[i] = 0;
		ones[i] = 1;
	}

	AccLibSvmSolver s;
	s.Solve(l, ONE_CLASS_Q(*prob,*param), zeros, ones,
		alpha, 1.0, 1.0, param->eps, si, param->shrinking);

	delete[] zeros;
	delete[] ones;
}

static void acclibsvm_solve_epsilon_svr(
	const acclibsvm_problem *prob, const svm_parameter *param,
	double *alpha, AccLibSvmSolver::SolutionInfo* si)
{
	int l = prob->l;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	for(i=0;i<l;i++)
	{
		alpha2[i] = 0;
		linear_term[i] = param->p - prob->y[i];
		y[i] = 1;

		alpha2[i+l] = 0;
		linear_term[i+l] = param->p + prob->y[i];
		y[i+l] = -1;
	}

	AccLibSvmSolver s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, param->C, param->C, param->eps, si, param->shrinking);

	double sum_alpha = 0;
	for(i=0;i<l;i++)
	{
		alpha[i] = alpha2[i] - alpha2[i+l];
		sum_alpha += fabs(alpha[i]);
	}
	info("nu = %f\n",sum_alpha/(param->C*l));

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

static void acclibsvm_solve_nu_svr(
	const acclibsvm_problem *prob, const svm_parameter *param,
	double *alpha, AccLibSvmSolver::SolutionInfo* si)
{
	int l = prob->l;
	double C = param->C;
	double *alpha2 = new double[2*l];
	double *linear_term = new double[2*l];
	schar *y = new schar[2*l];
	int i;

	double sum = C * param->nu * l / 2;
	for(i=0;i<l;i++)
	{
		alpha2[i] = alpha2[i+l] = min(sum,C);
		sum -= alpha2[i];

		linear_term[i] = - prob->y[i];
		y[i] = 1;

		linear_term[i+l] = prob->y[i];
		y[i+l] = -1;
	}

	AccLibSvmSolver_NU s;
	s.Solve(2*l, SVR_Q(*prob,*param), linear_term, y,
		alpha2, C, C, param->eps, si, param->shrinking);

	info("epsilon = %f\n",-si->r);

	for(i=0;i<l;i++)
		alpha[i] = alpha2[i] - alpha2[i+l];

	delete[] alpha2;
	delete[] linear_term;
	delete[] y;
}

//
// decision_function
//
struct decision_function
{
	double *alpha;
	double rho;	
};

static decision_function acclibsvm_train_one(
	const acclibsvm_problem *prob, const svm_parameter *param,
	double Cp, double Cn)
{
	double *alpha = Malloc(double,prob->l);
	AccLibSvmSolver::SolutionInfo si;
	switch(param->svm_type)
	{
		case C_SVC:
			acclibsvm_solve_c_svc(prob,param,alpha,&si,Cp,Cn);
			break;
		case NU_SVC:
			acclibsvm_solve_nu_svc(prob,param,alpha,&si);
			break;
		case ONE_CLASS:
			acclibsvm_solve_one_class(prob,param,alpha,&si);
			break;
		case EPSILON_SVR:
			acclibsvm_solve_epsilon_svr(prob,param,alpha,&si);
			break;
		case NU_SVR:
			acclibsvm_solve_nu_svr(prob,param,alpha,&si);
			break;
	}

	info("obj = %f, rho = %f\n",si.obj,si.rho);

	// output SVs

	int nSV = 0;
	int nBSV = 0;
	for(int i=0;i<prob->l;i++)
	{
		if(fabs(alpha[i]) > 0)
		{
			++nSV;
			if(prob->y[i] > 0)
			{
				if(fabs(alpha[i]) >= si.upper_bound_p)
					++nBSV;
			}
			else
			{
				if(fabs(alpha[i]) >= si.upper_bound_n)
					++nBSV;
			}
		}
	}

	info("nSV = %d, nBSV = %d\n",nSV,nBSV);

	decision_function f;
	f.alpha = alpha;
	f.rho = si.rho;
	return f;
}

// Platt's binary SVM Probablistic Output: an improvement from Lin et al.
static void sigmoid_train(
	int l, const double *dec_values, const double *labels, 
	double& A, double& B)
{
	double prior1=0, prior0 = 0;
	int i;

	for (i=0;i<l;i++)
		if (labels[i] > 0) prior1+=1;
		else prior0+=1;
	
	int max_iter=100;	// Maximal number of iterations
	double min_step=1e-10;	// Minimal step taken in line search
	double sigma=1e-12;	// For numerically strict PD of Hessian
	double eps=1e-5;
	double hiTarget=(prior1+1.0)/(prior1+2.0);
	double loTarget=1/(prior0+2.0);
	double *t=Malloc(double,l);
	double fApB,p,q,h11,h22,h21,g1,g2,det,dA,dB,gd,stepsize;
	double newA,newB,newf,d1,d2;
	int iter; 
	
	// Initial Point and Initial Fun Value
	A=0.0; B=log((prior0+1.0)/(prior1+1.0));
	double fval = 0.0;

	for (i=0;i<l;i++)
	{
		if (labels[i]>0) t[i]=hiTarget;
		else t[i]=loTarget;
		fApB = dec_values[i]*A+B;
		if (fApB>=0)
			fval += t[i]*fApB + log(1+exp(-fApB));
		else
			fval += (t[i] - 1)*fApB +log(1+exp(fApB));
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// Update Gradient and Hessian (use H' = H + sigma I)
		h11=sigma; // numerically ensures strict PD
		h22=sigma;
		h21=0.0;g1=0.0;g2=0.0;
		for (i=0;i<l;i++)
		{
			fApB = dec_values[i]*A+B;
			if (fApB >= 0)
			{
				p=exp(-fApB)/(1.0+exp(-fApB));
				q=1.0/(1.0+exp(-fApB));
			}
			else
			{
				p=1.0/(1.0+exp(fApB));
				q=exp(fApB)/(1.0+exp(fApB));
			}
			d2=p*q;
			h11+=dec_values[i]*dec_values[i]*d2;
			h22+=d2;
			h21+=dec_values[i]*d2;
			d1=t[i]-p;
			g1+=dec_values[i]*d1;
			g2+=d1;
		}

		// Stopping Criteria
		if (fabs(g1)<eps && fabs(g2)<eps)
			break;

		// Finding Newton direction: -inv(H') * g
		det=h11*h22-h21*h21;
		dA=-(h22*g1 - h21 * g2) / det;
		dB=-(-h21*g1+ h11 * g2) / det;
		gd=g1*dA+g2*dB;


		stepsize = 1;		// Line Search
		while (stepsize >= min_step)
		{
			newA = A + stepsize * dA;
			newB = B + stepsize * dB;

			// New function value
			newf = 0.0;
			for (i=0;i<l;i++)
			{
				fApB = dec_values[i]*newA+newB;
				if (fApB >= 0)
					newf += t[i]*fApB + log(1+exp(-fApB));
				else
					newf += (t[i] - 1)*fApB +log(1+exp(fApB));
			}
			// Check sufficient decrease
			if (newf<fval+0.0001*stepsize*gd)
			{
				A=newA;B=newB;fval=newf;
				break;
			}
			else
				stepsize = stepsize / 2.0;
		}

		if (stepsize < min_step)
		{
			info("Line search fails in two-class probability estimates\n");
			break;
		}
	}

	if (iter>=max_iter)
		info("Reaching maximal iterations in two-class probability estimates\n");
	free(t);
}

static double sigmoid_predict(double decision_value, double A, double B)
{
	double fApB = decision_value*A+B;
	// 1-p used later; avoid catastrophic cancellation
	if (fApB >= 0)
		return exp(-fApB)/(1.0+exp(-fApB));
	else
		return 1.0/(1+exp(fApB)) ;
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
static void multiclass_probability(int k, double **r, double *p)
{
	int t,j;
	int iter = 0, max_iter=max(100,k);
	double **Q=Malloc(double *,k);
	double *Qp=Malloc(double,k);
	double pQp, eps=0.005/k;
	
	for (t=0;t<k;t++)
	{
		p[t]=1.0/k;  // Valid if k = 1
		Q[t]=Malloc(double,k);
		Q[t][t]=0;
		for (j=0;j<t;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=Q[j][t];
		}
		for (j=t+1;j<k;j++)
		{
			Q[t][t]+=r[j][t]*r[j][t];
			Q[t][j]=-r[j][t]*r[t][j];
		}
	}
	for (iter=0;iter<max_iter;iter++)
	{
		// stopping condition, recalculate QP,pQP for numerical accuracy
		pQp=0;
		for (t=0;t<k;t++)
		{
			Qp[t]=0;
			for (j=0;j<k;j++)
				Qp[t]+=Q[t][j]*p[j];
			pQp+=p[t]*Qp[t];
		}
		double max_error=0;
		for (t=0;t<k;t++)
		{
			double error=fabs(Qp[t]-pQp);
			if (error>max_error)
				max_error=error;
		}
		if (max_error<eps) break;
		
		for (t=0;t<k;t++)
		{
			double diff=(-Qp[t]+pQp)/Q[t][t];
			p[t]+=diff;
			pQp=(pQp+diff*(diff*Q[t][t]+2*Qp[t]))/(1+diff)/(1+diff);
			for (j=0;j<k;j++)
			{
				Qp[j]=(Qp[j]+diff*Q[t][j])/(1+diff);
				p[j]/=(1+diff);
			}
		}
	}
	if (iter>=max_iter)
		info("Exceeds max_iter in multiclass_prob\n");
	for(t=0;t<k;t++) free(Q[t]);
	free(Q);
	free(Qp);
}

// Cross-validation decision values for probability estimates
static void acclibsvm_binary_svc_probability(
	const acclibsvm_problem *prob, const svm_parameter *param,
	double Cp, double Cn, double& probA, double& probB)
{
	int i;
	int nr_fold = 5;
	int *perm = Malloc(int,prob->l);
	double *dec_values = Malloc(double,prob->l);

	// random shuffle
	for(i=0;i<prob->l;i++) perm[i]=i;
	for(i=0;i<prob->l;i++)
	{
		int j = i+rand()%(prob->l-i);
		swap(perm[i],perm[j]);
	}
	for(i=0;i<nr_fold;i++)
	{
		int begin = i*prob->l/nr_fold;
		int end = (i+1)*prob->l/nr_fold;
		int j,k;
		struct acclibsvm_problem subprob;

		subprob.l = prob->l-(end-begin);
		subprob.x = Malloc(struct acclibsvm_node,subprob.l);
		subprob.y = Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<prob->l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		int p_count=0,n_count=0;
		for(j=0;j<k;j++)
			if(subprob.y[j]>0)
				p_count++;
			else
				n_count++;

		if(p_count==0 && n_count==0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 0;
		else if(p_count > 0 && n_count == 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = 1;
		else if(p_count == 0 && n_count > 0)
			for(j=begin;j<end;j++)
				dec_values[perm[j]] = -1;
		else
		{
			svm_parameter subparam = *param;
			subparam.probability=0;
			subparam.C=1.0;
			subparam.nr_weight=2;
			subparam.weight_label = Malloc(int,2);
			subparam.weight = Malloc(double,2);
			subparam.weight_label[0]=+1;
			subparam.weight_label[1]=-1;
			subparam.weight[0]=Cp;
			subparam.weight[1]=Cn;
			struct acclibsvm_model *submodel = acclibsvm_train(&subprob,&subparam);
			for(j=begin;j<end;j++)
			{
				acclibsvm_predict_values(submodel,(prob->x+perm[j]),&(dec_values[perm[j]])); 
				// ensure +1 -1 order; reason not using CV subroutine
				dec_values[perm[j]] *= submodel->label[0];
			}		
			acclibsvm_free_and_destroy_model(&submodel);
			svm_destroy_param(&subparam);
		}
		free(subprob.x);
		free(subprob.y);
	}		
	sigmoid_train(prob->l,dec_values,prob->y,probA,probB);
	free(dec_values);
	free(perm);
}

// Return parameter of a Laplace distribution 
static double acclibsvm_svr_probability(
	const acclibsvm_problem *prob, const svm_parameter *param)
{
	int i;
	int nr_fold = 5;
	double *ymv = Malloc(double,prob->l);
	double mae = 0;

	svm_parameter newparam = *param;
	newparam.probability = 0;
	acclibsvm_cross_validation(prob,&newparam,nr_fold,ymv);
	for(i=0;i<prob->l;i++)
	{
		ymv[i]=prob->y[i]-ymv[i];
		mae += fabs(ymv[i]);
	}		
	mae /= prob->l;
	double std=sqrt(2*mae*mae);
	int count=0;
	mae=0;
	for(i=0;i<prob->l;i++)
		if (fabs(ymv[i]) > 5*std) 
			count=count+1;
		else 
			mae+=fabs(ymv[i]);
	mae /= (prob->l-count);
	info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= %g\n",mae);
	free(ymv);
	return mae;
}


// label: label name, start: begin of each class, count: #data of classes, perm: indices to the original data
// perm, length l, must be allocated before calling this subroutine
static void acclibsvm_group_classes(const acclibsvm_problem *prob, int *nr_class_ret, int **label_ret, int **start_ret, int **count_ret, int *perm)
{
	int l = prob->l;
	int max_nr_class = 16;
	int nr_class = 0;
	int *label = Malloc(int,max_nr_class);
	int *count = Malloc(int,max_nr_class);
	int *data_label = Malloc(int,l);	
	int i;

	for(i=0;i<l;i++)
	{
		int this_label = (int)prob->y[i];
		int j;
		for(j=0;j<nr_class;j++)
		{
			if(this_label == label[j])
			{
				++count[j];
				break;
			}
		}
		data_label[i] = j;
		if(j == nr_class)
		{
			if(nr_class == max_nr_class)
			{
				max_nr_class *= 2;
				label = (int *)realloc(label,max_nr_class*sizeof(int));
				count = (int *)realloc(count,max_nr_class*sizeof(int));
			}
			label[nr_class] = this_label;
			count[nr_class] = 1;
			++nr_class;
		}
	}

	//
	// Labels are ordered by their first occurrence in the training set. 
	// However, for two-class sets with -1/+1 labels and -1 appears first, 
	// we swap labels to ensure that internally the binary SVM has positive data corresponding to the +1 instances.
	//
	if (nr_class == 2 && label[0] == -1 && label[1] == 1)
	{
		swap(label[0],label[1]);
		swap(count[0],count[1]);
		for(i=0;i<l;i++)
		{
			if(data_label[i] == 0)
				data_label[i] = 1;
			else
				data_label[i] = 0;
		}
	}
	
	int *start = Malloc(int,nr_class);
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];
	for(i=0;i<l;i++)
	{
		perm[start[data_label[i]]] = i;
		++start[data_label[i]];
	}
	start[0] = 0;
	for(i=1;i<nr_class;i++)
		start[i] = start[i-1]+count[i-1];

	*nr_class_ret = nr_class;
	*label_ret = label;
	*start_ret = start;
	*count_ret = count;
	free(data_label);
}

//
// Interface functions
//
acclibsvm_model *acclibsvm_train(const acclibsvm_problem *prob, const svm_parameter *param)
{
	acclibsvm_model *model = Malloc(acclibsvm_model,1);
	
	model->param = *param;
	model->free_sv = 0;	// XXX

	if(param->svm_type == ONE_CLASS ||
	   param->svm_type == EPSILON_SVR ||
	   param->svm_type == NU_SVR)
	{
		// regression or one-class-svm
		model->nr_class = 2;
		model->label = NULL;
		model->nSV = NULL;
		model->probA = NULL; model->probB = NULL;
		model->sv_coef = Malloc(double *,1);

		if(param->probability && 
		   (param->svm_type == EPSILON_SVR ||
		    param->svm_type == NU_SVR))
		{
			model->probA = Malloc(double,1);
			model->probA[0] = acclibsvm_svr_probability(prob,param);
		}

		decision_function f = acclibsvm_train_one(prob,param,0,0);
		model->rho = Malloc(double,1);
		model->rho[0] = f.rho;

		int nSV = 0;
		int i;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0) ++nSV;
		model->l = nSV;
		model->SV = Malloc(acclibsvm_node,nSV);
		model->sv_coef[0] = Malloc(double,nSV);
		model->sv_indices = Malloc(int,nSV);
		int j = 0;
		for(i=0;i<prob->l;i++)
			if(fabs(f.alpha[i]) > 0)
			{
				model->SV[j] = prob->x[i];
				model->sv_coef[0][j] = f.alpha[i];
				model->sv_indices[j] = i+1;
				++j;
			}		

		free(f.alpha);
	}
	else
	{
		// classification
		int l = prob->l;
		int nr_class;
		int *label = NULL;
		int *start = NULL;
		int *count = NULL;
		int *perm = Malloc(int,l);

		// group training data of the same class
		acclibsvm_group_classes(prob,&nr_class,&label,&start,&count,perm);		
		acclibsvm_node *x = Malloc(acclibsvm_node,l);
		int i;
		for(i=0;i<l;i++)
			x[i] = prob->x[perm[i]];

		// calculate weighted C

		double *weighted_C = Malloc(double, nr_class);
		for(i=0;i<nr_class;i++)
			weighted_C[i] = param->C;
		for(i=0;i<param->nr_weight;i++)
		{	
			int j;
			for(j=0;j<nr_class;j++)
				if(param->weight_label[i] == label[j])
					break;
			if(j == nr_class)
				fprintf(stderr,"warning: class label %d specified in weight is not found\n", param->weight_label[i]);
			else
				weighted_C[j] *= param->weight[i];
		}

		// train k*(k-1)/2 models
		
		bool *nonzero = Malloc(bool,l);
		for(i=0;i<l;i++)
			nonzero[i] = false;
		decision_function *f = Malloc(decision_function,nr_class*(nr_class-1)/2);

		double *probA=NULL,*probB=NULL;
		if (param->probability)
		{
			probA=Malloc(double,nr_class*(nr_class-1)/2);
			probB=Malloc(double,nr_class*(nr_class-1)/2);
		}

		int p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				acclibsvm_problem sub_prob;
				int si = start[i], sj = start[j];
				int ci = count[i], cj = count[j];
				sub_prob.l = ci+cj;
				sub_prob.x = Malloc(acclibsvm_node,sub_prob.l);
				sub_prob.y = Malloc(double,sub_prob.l);
				int k;
				for(k=0;k<ci;k++)
				{
					sub_prob.x[k] = x[si+k];
					sub_prob.y[k] = +1;
				}
				for(k=0;k<cj;k++)
				{
					sub_prob.x[ci+k] = x[sj+k];
					sub_prob.y[ci+k] = -1;
				}

				if(param->probability)
					acclibsvm_binary_svc_probability(&sub_prob,param,weighted_C[i],weighted_C[j],probA[p],probB[p]);

				f[p] = acclibsvm_train_one(&sub_prob,param,weighted_C[i],weighted_C[j]);
				for(k=0;k<ci;k++)
					if(!nonzero[si+k] && fabs(f[p].alpha[k]) > 0)
						nonzero[si+k] = true;
				for(k=0;k<cj;k++)
					if(!nonzero[sj+k] && fabs(f[p].alpha[ci+k]) > 0)
						nonzero[sj+k] = true;
				free(sub_prob.x);
				free(sub_prob.y);
				++p;
			}

		// build output

		model->nr_class = nr_class;
		
		model->label = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			model->label[i] = label[i];
		
		model->rho = Malloc(double,nr_class*(nr_class-1)/2);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			model->rho[i] = f[i].rho;

		if(param->probability)
		{
			model->probA = Malloc(double,nr_class*(nr_class-1)/2);
			model->probB = Malloc(double,nr_class*(nr_class-1)/2);
			for(i=0;i<nr_class*(nr_class-1)/2;i++)
			{
				model->probA[i] = probA[i];
				model->probB[i] = probB[i];
			}
		}
		else
		{
			model->probA=NULL;
			model->probB=NULL;
		}

		int total_sv = 0;
		int *nz_count = Malloc(int,nr_class);
		model->nSV = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
		{
			int nSV = 0;
			for(int j=0;j<count[i];j++)
				if(nonzero[start[i]+j])
				{	
					++nSV;
					++total_sv;
				}
			model->nSV[i] = nSV;
			nz_count[i] = nSV;
		}
		
		info("Total nSV = %d\n",total_sv);

		model->l = total_sv;
		model->SV = Malloc(acclibsvm_node,total_sv);
		model->sv_indices = Malloc(int,total_sv);
		p = 0;
		for(i=0;i<l;i++)
			if(nonzero[i])
			{
				model->SV[p] = x[i];
				model->sv_indices[p++] = perm[i] + 1;
			}
			
		int *nz_start = Malloc(int,nr_class);
		nz_start[0] = 0;
		for(i=1;i<nr_class;i++)
			nz_start[i] = nz_start[i-1]+nz_count[i-1];

		model->sv_coef = Malloc(double *,nr_class-1);
		for(i=0;i<nr_class-1;i++)
			model->sv_coef[i] = Malloc(double,total_sv);

		p = 0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				// classifier (i,j): coefficients with
				// i are in sv_coef[j-1][nz_start[i]...],
				// j are in sv_coef[i][nz_start[j]...]

				int si = start[i];
				int sj = start[j];
				int ci = count[i];
				int cj = count[j];
				
				int q = nz_start[i];
				int k;
				for(k=0;k<ci;k++)
					if(nonzero[si+k])
						model->sv_coef[j-1][q++] = f[p].alpha[k];
				q = nz_start[j];
				for(k=0;k<cj;k++)
					if(nonzero[sj+k])
						model->sv_coef[i][q++] = f[p].alpha[ci+k];
				++p;
			}
		
		free(label);
		free(probA);
		free(probB);
		free(count);
		free(perm);
		free(start);
		free(x);
		free(weighted_C);
		free(nonzero);
		for(i=0;i<nr_class*(nr_class-1)/2;i++)
			free(f[i].alpha);
		free(f);
		free(nz_count);
		free(nz_start);
	}
	return model;
}

// Stratified cross validation
void acclibsvm_cross_validation(const acclibsvm_problem *prob, const svm_parameter *param, int nr_fold, double *target)
{
	int i;
	int *fold_start;
	int l = prob->l;
	int *perm = Malloc(int,l);
	int nr_class;
	if (nr_fold > l)
	{
		nr_fold = l;
		fprintf(stderr,"WARNING: # folds > # data. Will use # folds = # data instead (i.e., leave-one-out cross validation)\n");
	}
	fold_start = Malloc(int,nr_fold+1);
	// stratified cv may not give leave-one-out rate
	// Each class to l folds -> some folds may have zero elements
	if((param->svm_type == C_SVC ||
	    param->svm_type == NU_SVC) && nr_fold < l)
	{
		int *start = NULL;
		int *label = NULL;
		int *count = NULL;
		acclibsvm_group_classes(prob,&nr_class,&label,&start,&count,perm);

		// random shuffle and then data grouped by fold using the array perm
		int *fold_count = Malloc(int,nr_fold);
		int c;
		int *index = Malloc(int,l);
		for(i=0;i<l;i++)
			index[i]=perm[i];
		for (c=0; c<nr_class; c++) 
			for(i=0;i<count[c];i++)
			{
				int j = i+rand()%(count[c]-i);
				swap(index[start[c]+j],index[start[c]+i]);
			}
		for(i=0;i<nr_fold;i++)
		{
			fold_count[i] = 0;
			for (c=0; c<nr_class;c++)
				fold_count[i]+=(i+1)*count[c]/nr_fold-i*count[c]/nr_fold;
		}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		for (c=0; c<nr_class;c++)
			for(i=0;i<nr_fold;i++)
			{
				int begin = start[c]+i*count[c]/nr_fold;
				int end = start[c]+(i+1)*count[c]/nr_fold;
				for(int j=begin;j<end;j++)
				{
					perm[fold_start[i]] = index[j];
					fold_start[i]++;
				}
			}
		fold_start[0]=0;
		for (i=1;i<=nr_fold;i++)
			fold_start[i] = fold_start[i-1]+fold_count[i-1];
		free(start);	
		free(label);
		free(count);	
		free(index);
		free(fold_count);
	}
	else
	{
		for(i=0;i<l;i++) perm[i]=i;
		for(i=0;i<l;i++)
		{
			int j = i+rand()%(l-i);
			swap(perm[i],perm[j]);
		}
		for(i=0;i<=nr_fold;i++)
			fold_start[i]=i*l/nr_fold;
	}

	for(i=0;i<nr_fold;i++)
	{
		int begin = fold_start[i];
		int end = fold_start[i+1];
		int j,k;
		struct acclibsvm_problem subprob;

		subprob.l = l-(end-begin);
		subprob.x = Malloc(struct acclibsvm_node,subprob.l);
		subprob.y = Malloc(double,subprob.l);
			
		k=0;
		for(j=0;j<begin;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		for(j=end;j<l;j++)
		{
			subprob.x[k] = prob->x[perm[j]];
			subprob.y[k] = prob->y[perm[j]];
			++k;
		}
		struct acclibsvm_model *submodel = acclibsvm_train(&subprob,param);
		if(param->probability && 
		   (param->svm_type == C_SVC || param->svm_type == NU_SVC))
		{
			double *prob_estimates=Malloc(double,acclibsvm_get_nr_class(submodel));
			for(j=begin;j<end;j++)
				target[perm[j]] = acclibsvm_predict_probability(submodel,(prob->x + perm[j]),prob_estimates);
			free(prob_estimates);			
		}
		else
			for(j=begin;j<end;j++)
				target[perm[j]] = acclibsvm_predict(submodel,prob->x+perm[j]);
		acclibsvm_free_and_destroy_model(&submodel);
		free(subprob.x);
		free(subprob.y);
	}		
	free(fold_start);
	free(perm);	
}


int acclibsvm_get_svm_type(const acclibsvm_model *model)
{
	return model->param.svm_type;
}

int acclibsvm_get_nr_class(const acclibsvm_model *model)
{
	return model->nr_class;
}

void acclibsvm_get_labels(const acclibsvm_model *model, int* label)
{
	if (model->label != NULL)
		for(int i=0;i<model->nr_class;i++)
			label[i] = model->label[i];
}

void acclibsvm_get_sv_indices(const acclibsvm_model *model, int* indices)
{
	if (model->sv_indices != NULL)
		for(int i=0;i<model->l;i++)
			indices[i] = model->sv_indices[i];
}

int acclibsvm_get_nr_sv(const acclibsvm_model *model)
{
	return model->l;
}

double acclibsvm_get_svr_probability(const acclibsvm_model *model)
{
	if ((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
	    model->probA!=NULL)
		return model->probA[0];
	else
	{
		fprintf(stderr,"Model doesn't contain information for SVR probability inference\n");
		return 0;
	}
}

double acclibsvm_predict_values(const acclibsvm_model *model, const acclibsvm_node *x, double* dec_values)
{
	int i;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
	{
		double *sv_coef = model->sv_coef[0];
		double sum = 0;
		
		Qfloat *kvalue = Malloc(Qfloat,model->l);
		//for(i=0;i<model->l;i++)
		//	sum += sv_coef[i] * AccLibSvmKernel::k_function(x,model->SV+i,model->param);
		//AccLibSvmKernel::kernel_function_vect(kvalue, x, model->SV, 0, model->l, model->param);
		for(i=0;i<model->l;i++) sum += sv_coef[i] * kvalue[i];

		free(kvalue);

		sum -= model->rho[0];
		*dec_values = sum;
		

		if(model->param.svm_type == ONE_CLASS)
			return (sum>0)?1:-1;
		else
			return sum;
	}
	else
	{
		int i;
		int nr_class = model->nr_class;
		int l = model->l;
		
		Qfloat *kvalue = Malloc(Qfloat,l);
		//for(i=0;i<l;i++)
		//	kvalue[i] = AccLibSvmKernel::k_function(x,model->SV+i,model->param);
		//AccLibSvmKernel::kernel_function_vect(kvalue, x, model->SV, 0, l, model->param);

		int *start = Malloc(int,nr_class);
		start[0] = 0;
		for(i=1;i<nr_class;i++)
			start[i] = start[i-1]+model->nSV[i-1];

		int *vote = Malloc(int,nr_class);
		for(i=0;i<nr_class;i++)
			vote[i] = 0;

		int p=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				double sum = 0;
				int si = start[i];
				int sj = start[j];
				int ci = model->nSV[i];
				int cj = model->nSV[j];
				
				int k;
				double *coef1 = model->sv_coef[j-1];
				double *coef2 = model->sv_coef[i];
				for(k=0;k<ci;k++)
					sum += coef1[si+k] * kvalue[si+k];
				for(k=0;k<cj;k++)
					sum += coef2[sj+k] * kvalue[sj+k];
				sum -= model->rho[p];
				dec_values[p] = sum;

				if(dec_values[p] > 0)
					++vote[i];
				else
					++vote[j];
				p++;
			}

		int vote_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(vote[i] > vote[vote_max_idx])
				vote_max_idx = i;

		free(kvalue);
		free(start);
		free(vote);
		return model->label[vote_max_idx];
	}
}

double acclibsvm_predict(const acclibsvm_model *model, const acclibsvm_node *x)
{
	int nr_class = model->nr_class;
	double *dec_values;
	if(model->param.svm_type == ONE_CLASS ||
	   model->param.svm_type == EPSILON_SVR ||
	   model->param.svm_type == NU_SVR)
		dec_values = Malloc(double, 1);
	else 
		dec_values = Malloc(double, nr_class*(nr_class-1)/2);
	double pred_result = acclibsvm_predict_values(model, x, dec_values);
	free(dec_values);
	return pred_result;
}

double acclibsvm_predict_probability(
	const acclibsvm_model *model, const acclibsvm_node *x, double *prob_estimates)
{
	if ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
	    model->probA!=NULL && model->probB!=NULL)
	{
		int i;
		int nr_class = model->nr_class;
		double *dec_values = Malloc(double, nr_class*(nr_class-1)/2);
		acclibsvm_predict_values(model, x, dec_values);

		double min_prob=1e-7;
		double **pairwise_prob=Malloc(double *,nr_class);
		for(i=0;i<nr_class;i++)
			pairwise_prob[i]=Malloc(double,nr_class);
		int k=0;
		for(i=0;i<nr_class;i++)
			for(int j=i+1;j<nr_class;j++)
			{
				pairwise_prob[i][j]=min(max(sigmoid_predict(dec_values[k],model->probA[k],model->probB[k]),min_prob),1-min_prob);
				pairwise_prob[j][i]=1-pairwise_prob[i][j];
				k++;
			}
		multiclass_probability(nr_class,pairwise_prob,prob_estimates);

		int prob_max_idx = 0;
		for(i=1;i<nr_class;i++)
			if(prob_estimates[i] > prob_estimates[prob_max_idx])
				prob_max_idx = i;
		for(i=0;i<nr_class;i++)
			free(pairwise_prob[i]);
		free(dec_values);
		free(pairwise_prob);	     
		return model->label[prob_max_idx];
	}
	else 
		return acclibsvm_predict(model, x);
}

static const char *svm_type_table[] =
{
	"c_svc","nu_svc","one_class","epsilon_svr","nu_svr",NULL
};

static const char *kernel_type_table[]=
{
	"linear","polynomial","rbf","sigmoid","precomputed",NULL
};

int acclibsvm_save_model(const char *model_file_name, const acclibsvm_model *model, int dimOffset)
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");
	
	const svm_parameter& param = model->param;

	fprintf(fp,"svm_type %s\n", svm_type_table[param.svm_type]);
	fprintf(fp,"kernel_type %s\n", kernel_type_table[param.kernel_type]);

	if(param.kernel_type == POLY)
		fprintf(fp,"degree %d\n", param.degree);

	if(param.kernel_type == POLY || param.kernel_type == RBF || param.kernel_type == SIGMOID)
		fprintf(fp,"gamma %g\n", param.gamma);

	if(param.kernel_type == POLY || param.kernel_type == SIGMOID)
		fprintf(fp,"coef0 %g\n", param.coef0);

	int nr_class = model->nr_class;
	int l = model->l;
	fprintf(fp, "nr_class %d\n", nr_class);
	fprintf(fp, "total_sv %d\n",l);
	
	{
		fprintf(fp, "rho");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->rho[i]);
		fprintf(fp, "\n");
	}
	
	if(model->label)
	{
		fprintf(fp, "label");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->label[i]);
		fprintf(fp, "\n");
	}

	if(model->probA) // regression has probA only
	{
		fprintf(fp, "probA");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probA[i]);
		fprintf(fp, "\n");
	}
	if(model->probB)
	{
		fprintf(fp, "probB");
		for(int i=0;i<nr_class*(nr_class-1)/2;i++)
			fprintf(fp," %g",model->probB[i]);
		fprintf(fp, "\n");
	}

	if(model->nSV)
	{
		fprintf(fp, "nr_sv");
		for(int i=0;i<nr_class;i++)
			fprintf(fp," %d",model->nSV[i]);
		fprintf(fp, "\n");
	}

	fprintf(fp, "SV\n");
	const double * const *sv_coef = model->sv_coef;
	const acclibsvm_node *SV = model->SV;

	for(int i=0;i<l;i++)
	{
		for(int j=0;j<nr_class-1;j++)
			fprintf(fp, "%.16g ",sv_coef[j][i]);

		const acclibsvm_node *p = (SV + i);

		if(param.kernel_type == PRECOMPUTED)
			fprintf(fp,"0:%d ",(int)(p->values[0]));
		else
			for (int j = 0; j < p->num; j++)
				if (p->values[j] != 0.0)
					if(p->ind == NULL) //DENSE
						fprintf(fp,"%d:%.8g ", j + dimOffset, p->values[j]);
					else
						fprintf(fp,"%d:%.8g ", p->ind[j] + dimOffset, p->values[j]);
		fprintf(fp, "\n");
	}
	
	setlocale(LC_ALL, old_locale);
	free(old_locale);
	
	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

//static char *line = NULL;
//static int max_line_len;

//static char* readline(FILE *input)
//{
//	int len;
//
//	if(fgets(line,max_line_len,input) == NULL)
//		return NULL;
//
//	while(strrchr(line,'\n') == NULL)
//	{
//		max_line_len *= 2;
//		line = (char *) realloc(line,max_line_len);
//		len = (int) strlen(line);
//		if(fgets(line+len,max_line_len-len,input) == NULL)
//			break;
//	}
//	return line;
//}
//
//svm_model *svm_load_model(const char *model_file_name)
//{
//	FILE *fp = fopen(model_file_name,"rb");
//	if(fp==NULL) return NULL;
//	
//	char *old_locale = strdup(setlocale(LC_ALL, NULL));
//	setlocale(LC_ALL, "C");
//	
//	// read parameters
//
//	svm_model *model = Malloc(svm_model,1);
//	svm_parameter& param = model->param;
//	model->rho = NULL;
//	model->probA = NULL;
//	model->probB = NULL;
//	model->sv_indices = NULL;
//	model->label = NULL;
//	model->nSV = NULL;
//
//	char cmd[81];
//	while(1)
//	{
//		fscanf(fp,"%80s",cmd);
//
//		if(strcmp(cmd,"svm_type")==0)
//		{
//			fscanf(fp,"%80s",cmd);
//			int i;
//			for(i=0;svm_type_table[i];i++)
//			{
//				if(strcmp(svm_type_table[i],cmd)==0)
//				{
//					param.svm_type=i;
//					break;
//				}
//			}
//			if(svm_type_table[i] == NULL)
//			{
//				fprintf(stderr,"unknown svm type.\n");
//				
//				setlocale(LC_ALL, old_locale);
//				free(old_locale);
//				free(model->rho);
//				free(model->label);
//				free(model->nSV);
//				free(model);
//				return NULL;
//			}
//		}
//		else if(strcmp(cmd,"kernel_type")==0)
//		{		
//			fscanf(fp,"%80s",cmd);
//			int i;
//			for(i=0;kernel_type_table[i];i++)
//			{
//				if(strcmp(kernel_type_table[i],cmd)==0)
//				{
//					param.kernel_type=i;
//					break;
//				}
//			}
//			if(kernel_type_table[i] == NULL)
//			{
//				fprintf(stderr,"unknown kernel function.\n");
//				
//				setlocale(LC_ALL, old_locale);
//				free(old_locale);
//				free(model->rho);
//				free(model->label);
//				free(model->nSV);
//				free(model);
//				return NULL;
//			}
//		}
//		else if(strcmp(cmd,"degree")==0)
//			fscanf(fp,"%d",&param.degree);
//		else if(strcmp(cmd,"gamma")==0)
//			fscanf(fp,"%lf",&param.gamma);
//		else if(strcmp(cmd,"coef0")==0)
//			fscanf(fp,"%lf",&param.coef0);
//		else if(strcmp(cmd,"nr_class")==0)
//			fscanf(fp,"%d",&model->nr_class);
//		else if(strcmp(cmd,"total_sv")==0)
//			fscanf(fp,"%d",&model->l);
//		else if(strcmp(cmd,"rho")==0)
//		{
//			int n = model->nr_class * (model->nr_class-1)/2;
//			model->rho = Malloc(double,n);
//			for(int i=0;i<n;i++)
//				fscanf(fp,"%lf",&model->rho[i]);
//		}
//		else if(strcmp(cmd,"label")==0)
//		{
//			int n = model->nr_class;
//			model->label = Malloc(int,n);
//			for(int i=0;i<n;i++)
//				fscanf(fp,"%d",&model->label[i]);
//		}
//		else if(strcmp(cmd,"probA")==0)
//		{
//			int n = model->nr_class * (model->nr_class-1)/2;
//			model->probA = Malloc(double,n);
//			for(int i=0;i<n;i++)
//				fscanf(fp,"%lf",&model->probA[i]);
//		}
//		else if(strcmp(cmd,"probB")==0)
//		{
//			int n = model->nr_class * (model->nr_class-1)/2;
//			model->probB = Malloc(double,n);
//			for(int i=0;i<n;i++)
//				fscanf(fp,"%lf",&model->probB[i]);
//		}
//		else if(strcmp(cmd,"nr_sv")==0)
//		{
//			int n = model->nr_class;
//			model->nSV = Malloc(int,n);
//			for(int i=0;i<n;i++)
//				fscanf(fp,"%d",&model->nSV[i]);
//		}
//		else if(strcmp(cmd,"SV")==0)
//		{
//			while(1)
//			{
//				int c = getc(fp);
//				if(c==EOF || c=='\n') break;	
//			}
//			break;
//		}
//		else
//		{
//			fprintf(stderr,"unknown text in model file: [%s]\n",cmd);
//			
//			setlocale(LC_ALL, old_locale);
//			free(old_locale);
//			free(model->rho);
//			free(model->label);
//			free(model->nSV);
//			free(model);
//			return NULL;
//		}
//	}
//
//	// read sv_coef and SV
//
//	int elements = 0;
//	long pos = ftell(fp);
//
//	max_line_len = 1024;
//	line = Malloc(char,max_line_len);
//	char *p,*endptr,*idx,*val;
//
//#ifdef _DENSE_REP
//	int max_index = 1;
//	// read the max dimension of all vectors
//	while(readline(fp) != NULL)
//	{
//		char *p;
//		p = strrchr(line, ':');
//		if(p != NULL)
//		{			
//			while(*p != ' ' && *p != '\t' && p > line)
//				p--;
//			if(p > line)
//				max_index = (int) strtol(p,&endptr,10) + 1;
//		}		
//		if(max_index > elements)
//			elements = max_index;
//	}
//#else
//	while(readline(fp)!=NULL)
//	{
//		p = strtok(line,":");
//		while(1)
//		{
//			p = strtok(NULL,":");
//			if(p == NULL)
//				break;
//			++elements;
//		}
//	}
//	elements += model->l;
//
//#endif
//	fseek(fp,pos,SEEK_SET);
//
//	int m = model->nr_class - 1;
//	int l = model->l;
//	model->sv_coef = Malloc(double *,m);
//	int i;
//	for(i=0;i<m;i++)
//		model->sv_coef[i] = Malloc(double,l);
//
//#ifdef _DENSE_REP
//	int index;
//	model->SV = Malloc(svm_node,l);
//
//	for(i=0;i<l;i++)
//	{
//		readline(fp);
//
//		model->SV[i].values = Malloc(double, elements);
//		model->SV[i].dim = 0;
//
//		p = strtok(line, " \t");
//		model->sv_coef[0][i] = strtod(p,&endptr);
//		for(int k=1;k<m;k++)
//		{
//			p = strtok(NULL, " \t");
//			model->sv_coef[k][i] = strtod(p,&endptr);
//		}
//
//		int *d = &(model->SV[i].dim);
//		while(1)
//		{
//			idx = strtok(NULL, ":");
//			val = strtok(NULL, " \t");
//
//			if(val == NULL)
//				break;
//			index = (int) strtol(idx,&endptr,10);
//			while (*d < index)
//				model->SV[i].values[(*d)++] = 0.0;
//			model->SV[i].values[(*d)++] = strtod(val,&endptr);
//		}
//	}
//#else
//	model->SV = Malloc(svm_node*,l);
//	svm_node *x_space = NULL;
//	if(l>0) x_space = Malloc(svm_node,elements);
//	int j=0;
//	for(i=0;i<l;i++)
//	{
//		readline(fp);
//		model->SV[i] = &x_space[j];
//
//		p = strtok(line, " \t");
//		model->sv_coef[0][i] = strtod(p,&endptr);
//		for(int k=1;k<m;k++)
//		{
//			p = strtok(NULL, " \t");
//			model->sv_coef[k][i] = strtod(p,&endptr);
//		}
//
//		while(1)
//		{
//			idx = strtok(NULL, ":");
//			val = strtok(NULL, " \t");
//
//			if(val == NULL)
//				break;
//			x_space[j].index = (int) strtol(idx,&endptr,10);
//			x_space[j].value = strtod(val,&endptr);
//
//			++j;
//		}
//		x_space[j++].index = -1;
//	}
//#endif
//	free(line);
//
//	setlocale(LC_ALL, old_locale);
//	free(old_locale);
//	
//	if (ferror(fp) != 0 || fclose(fp) != 0)
//		return NULL;
//
//	model->free_sv = 1;	// XXX
//	return model;
//}

void acclibsvm_free_model_content(acclibsvm_model* model_ptr)
{
	if(model_ptr->free_sv && model_ptr->l > 0 && model_ptr->SV != NULL)
	for (int i = 0; i < model_ptr->l; i++)
		free (model_ptr->SV[i].values);
	if(model_ptr->sv_coef)
	{
		for(int i=0;i<model_ptr->nr_class-1;i++)
			free(model_ptr->sv_coef[i]);
	}
	free(model_ptr->SV);
	model_ptr->SV = NULL;

	free(model_ptr->sv_coef);
	model_ptr->sv_coef = NULL;

	free(model_ptr->rho);
	model_ptr->rho = NULL;

	free(model_ptr->label);
	model_ptr->label= NULL;

	free(model_ptr->probA);
	model_ptr->probA = NULL;

	free(model_ptr->probB);
	model_ptr->probB= NULL;

	free(model_ptr->sv_indices);
	model_ptr->sv_indices = NULL;

	free(model_ptr->nSV);
	model_ptr->nSV = NULL;
}

void acclibsvm_free_and_destroy_model(acclibsvm_model** model_ptr_ptr)
{
	if(model_ptr_ptr != NULL && *model_ptr_ptr != NULL)
	{
		acclibsvm_free_model_content(*model_ptr_ptr);
		free(*model_ptr_ptr);
		*model_ptr_ptr = NULL;
	}
}

void acclibsvm_destroy_param(svm_parameter* param)
{
	free(param->weight_label);
	free(param->weight);
}

const char *acclibsvm_check_parameter(const svm_problem *prob, const svm_parameter *param)
{
	// svm_type

	int svm_type = param->svm_type;
	if(svm_type != C_SVC &&
	   svm_type != NU_SVC &&
	   svm_type != ONE_CLASS &&
	   svm_type != EPSILON_SVR &&
	   svm_type != NU_SVR)
		return "unknown svm type";
	
	// kernel_type, degree
	
	int kernel_type = param->kernel_type;
	if(kernel_type != LINEAR &&
	   kernel_type != POLY &&
	   kernel_type != RBF &&
	   kernel_type != SIGMOID &&
	   kernel_type != PRECOMPUTED)
		return "unknown kernel type";

	if(param->gamma < 0)
		return "gamma < 0";

	if(param->degree < 0)
		return "degree of polynomial kernel < 0";

	// cache_size,eps,C,nu,p,shrinking

	if(param->cache_size <= 0)
		return "cache_size <= 0";

	if(param->eps <= 0)
		return "eps <= 0";

	if(svm_type == C_SVC ||
	   svm_type == EPSILON_SVR ||
	   svm_type == NU_SVR)
		if(param->C <= 0)
			return "C <= 0";

	if(svm_type == NU_SVC ||
	   svm_type == ONE_CLASS ||
	   svm_type == NU_SVR)
		if(param->nu <= 0 || param->nu > 1)
			return "nu <= 0 or nu > 1";

	if(svm_type == EPSILON_SVR)
		if(param->p < 0)
			return "p < 0";

	if(param->shrinking != 0 &&
	   param->shrinking != 1)
		return "shrinking != 0 and shrinking != 1";

	if(param->probability != 0 &&
	   param->probability != 1)
		return "probability != 0 and probability != 1";

	if(param->probability == 1 &&
	   svm_type == ONE_CLASS)
		return "one-class SVM probability output not supported yet";


	// check whether nu-svc is feasible
	
	if(svm_type == NU_SVC)
	{
		int l = prob->l;
		int max_nr_class = 16;
		int nr_class = 0;
		int *label = Malloc(int,max_nr_class);
		int *count = Malloc(int,max_nr_class);

		int i;
		for(i=0;i<l;i++)
		{
			int this_label = (int)prob->y[i];
			int j;
			for(j=0;j<nr_class;j++)
				if(this_label == label[j])
				{
					++count[j];
					break;
				}
			if(j == nr_class)
			{
				if(nr_class == max_nr_class)
				{
					max_nr_class *= 2;
					label = (int *)realloc(label,max_nr_class*sizeof(int));
					count = (int *)realloc(count,max_nr_class*sizeof(int));
				}
				label[nr_class] = this_label;
				count[nr_class] = 1;
				++nr_class;
			}
		}
	
		for(i=0;i<nr_class;i++)
		{
			int n1 = count[i];
			for(int j=i+1;j<nr_class;j++)
			{
				int n2 = count[j];
				if(param->nu*(n1+n2)/2 > min(n1,n2))
				{
					free(label);
					free(count);
					return "specified nu is infeasible";
				}
			}
		}
		free(label);
		free(count);
	}

	return NULL;
}

int acclibsvm_check_probability_model(const acclibsvm_model *model)
{
	return ((model->param.svm_type == C_SVC || model->param.svm_type == NU_SVC) &&
		model->probA!=NULL && model->probB!=NULL) ||
		((model->param.svm_type == EPSILON_SVR || model->param.svm_type == NU_SVR) &&
		 model->probA!=NULL);
}

void acclibsvm_set_print_string_function(void (*print_func)(const char *))
{
	if(print_func == NULL)
		svm_print_string = &print_string_stdout;
	else
		svm_print_string = print_func;
}