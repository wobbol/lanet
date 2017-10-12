// $cc main.c -l cblas -l m -o lanet
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cblas.h>

struct matrix
{
	char *name;
	int height;
	int width;
	float d[100];
};

void print_fmatrix(const struct matrix *const c)
{
		puts(c->name);
	for(int i = 0; i < c->height; ++i){
		for(int j = 0; j < c->width; ++j){
			printf("%f ",c->d[c->width*i+j]);
		}
		puts("");
	}
}

void mf_apply(struct matrix *const m, float (*f)(float))
{
	float *tmp;
	for(int i = 0; i < m->height; ++i){
		for(int j = 0; j < m->width; ++j){
			tmp = &m->d[m->width*i+j];
			*tmp = f(*tmp);
		}
	}
}

float sigmoid(float in)
{
	return 1.0/(1+exp(-in));
}

void copy(
const struct matrix *const src,
struct matrix *const dest)
{
	//assert(sizeof src.d <= sizeof dest.d)
	cblas_scopy(src->height,src->d,1,dest->d,1);
	dest->height = src->height;
	dest->width = src->width;
}

void ambpc(
const struct matrix *const a,
const struct matrix *const b,
const struct matrix *const c, 
struct matrix *const out)
{

	// blas gemm: C <- {alpha}AB + {beta}C
	copy(c, out);

	//no input data overflows struct matrix
	assert(a->height*a->width < 100);
	assert(b->height*b->width < 100);
	assert(c->height*c->width < 100);

	//no output data overflows struct matrix
	assert(a->height*b->width < 100);

	//c dimensions match result of matrix multiplication
	assert(out->height == a->height);
	assert(out->width == b->width);


	int m = a->height;//height first
	int k = b->width;//width second
	int n = b->height;//shared dimension

	int lda= k;
	int ldb= n;
	int ldc= n;

	CBLAS_LAYOUT layout = CblasRowMajor;
	CBLAS_TRANSPOSE trans_a = CblasNoTrans;
	CBLAS_TRANSPOSE trans_b = CblasNoTrans;

	//assert(weight_h.width == act_in.height)
	cblas_sgemm(layout, trans_a, trans_b, 
		m, n, k,
                 1,
		 a->d, lda,
		 b->d, ldb,
		 -1,
		 out->d, ldc);
}

void addr(const void *const in){
	printf("address of: %u\n",(unsigned int)&in);
}

int main(void)
{
	float alpha = 1;
	float beta = 1;
	const struct matrix act_in = {
		.name = "activation in",
		.height = 4,
		.width = 1,
		.d = {
			.3,
			.5,
			.2,
			.1},
	};

	const struct matrix weight_h = {
		.name = "weight hidden",
		.height = 3,
		.width = 4,
		.d = {
			.4,.5,.3,.5,
			.5,.1,.3,.2,
			.5,.2,.3,.9},
	};
	const struct matrix bias_h = {
		.name = "bias hidden",
		.height = 3,
		.width = 1,
		.d = {
			1,
			4,
			6},
	};

	struct matrix act_h = {
		.name = "activation hidden",
		.height = 3,
		.width = 1,
		.d = {
			1,
			1,
			1},
	};

	const struct matrix weight_o = {
		.name = "weight output",
		.height = 5,
		.width = 3,
		.d = {
			.4,.5,.3,
			.5,.3,.5,
			.1,.3,.7,
			.3,.6,.3,
			.1,.6,.1},
	};

	const struct matrix bias_o = {
		.name = "bias output",
		.height = 5,
		.width = 1,
		.d = {.4,.6,.3,.1,.7},
	};

	struct matrix c = {
		.name = "c",
		0,
	};

	//calc z
	ambpc(&weight_h,&act_in,&bias_h,&act_h);

	//calc activation
	mf_apply(&act_h,sigmoid);

	//calc z
	ambpc(&weight_o,&act_h,&bias_o,&c);
	//calc_activation

	mf_apply(&c,sigmoid);
	//print

	print_fmatrix(&c);

	return 0;
}
