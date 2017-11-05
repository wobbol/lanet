// $cc main.c -l cblas -l m -o lanet
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include <time.h>
#include <math.h>
#include <cblas.h>

struct matrix;
struct layer;
struct net;

struct matrix
{
	char *name;
	const struct layer *from;
	int height;
	int width;
	CBLAS_TRANSPOSE t;
	float d[100];
};

struct layer
{
	char *name;
	int length;
	struct matrix weight;
	struct matrix bias;
	struct matrix weight_error;
	struct matrix bias_error;
	struct matrix z;
	struct matrix act;
	struct layer *in;
	struct layer *out;
};

struct net
{
	char *name;
	struct layer *top;
};


void print_fmatrix(const struct matrix *const c)
{
		if(c->from && c->from->name)
			printf("l_name: %s\n",c->from->name);
		if(c->name)
			printf("m_name: %s\n",c->name);
	for(int i = 0; i < c->height; ++i){
		for(int j = 0; j < c->width; ++j){
			printf("%f ",c->d[c->width*i+j]);
		}
		puts("");
	}
}

int print_layer(const struct layer *const l)
{
	printf("\nl_name: %s\n",l->name);
	printf("num: %d\n", l->length);
	print_fmatrix(&l->weight);
	print_fmatrix(&l->bias);
	print_fmatrix(&l->weight_error);
	print_fmatrix(&l->bias_error);
	print_fmatrix(&l->z);
	print_fmatrix(&l->act);
	return l->in ? 1:0;

}

void print_net(const struct net *const n)
{
	struct layer *tmp = n->top;
	while(print_layer(tmp))
		tmp = tmp->in;
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
float sigmoid_prime(float in)
{
	return exp(in)/pow(1 + exp(in),2);
}

void copy(
const struct matrix *const src,
struct matrix *const dest)
{
	//assert(sizeof src.d <= sizeof dest.d)
	cblas_scopy(src->height * src->width,src->d,1,dest->d,1);
	dest->height = src->height;
	dest->width = src->width;
	dest->t = src->t;
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


	int rows_a_c = a->height;
	int cols_b_c = b->width;
	int cols_a_rows_b = b->width;

	int lda= a->width;
	int ldb= b->width;
	int ldc= c->width;

	CBLAS_LAYOUT layout = CblasRowMajor;

	//assert(weight_h.width == act_in.height)
	cblas_sgemm(layout, a->t, b->t,
		rows_a_c, cols_b_c, cols_a_rows_b,
                 1,
		 a->d, lda,
		 b->d, ldb,
		 1,
		 out->d, ldc);
}

void hadamard(
const struct matrix *const a,
const struct matrix *const b,
struct matrix *const out)
{
	assert((a->height * a->width) == (b->height * b->width));
	copy(b,out);

	int loop = a->height * a->width;
	for(int i = 0; i < loop; ++i)
		out->d[i] *= a->d[i];

	return;
}
void error(
struct matrix *const w,
const struct matrix *const er,
const struct matrix *const z,
struct matrix *const out)
{

	w->t = CblasTrans;
	ambpc(w,er,z,out);
	w->t = CblasNoTrans;
	return;
}
void first_error(
const struct matrix *const a,
const struct matrix *const z,
const struct matrix *const ex,
struct matrix *const out)
{
	assert(a->width == 1);
	assert(z->width == 1);

	assert(a->height == z->height);

	//gradient of cost function wrt to activation
	// using quadratic cost function
	// 2 * (expected - a) * sigmoid_prime(z)
	int loop = a->height;
	for(int i = 0; i < loop; ++i){
		out->d[i] = 2 * (ex->d[i] - a->d[i]) * sigmoid_prime(z->d[i]);

	}
	return;
}


void addr(const void *const in){
	printf("address of: %u\n",(unsigned int)&in);
}

float gen_rand(void)
{
	return (float)rand()/RAND_MAX;
}

float gen_zero(void)
{
	return 0.0;
}
float gen_one(void)
{
	return 1.0;
}

void init_matrix(
struct matrix *const ret,
const int height,
const int width,
float(*gen)(void),
const struct layer *from,
const char *const name)
{
	char *str = malloc(strlen(name)+1);
	strcpy(str,name);
	ret->name = str;

	ret->from = from;
	ret->height = height;
	ret->width = width;
	ret->t = CblasNoTrans;
	for(int i = 0; i < height * width; ++i)
		ret->d[i] = gen();
}

struct layer *init_layer(
const int length,
const char *const name,
struct layer *const in) // requires NULL if input layer.
{
	struct layer *ret = malloc(sizeof(*ret));

	char *str = malloc(strlen(name));
	strcpy(str,name);
	ret->name = str;

	ret->length = length;

	init_matrix(&ret->z,length,1,gen_zero,ret,"weighted sum");
	init_matrix(&ret->act,length,1,gen_one,ret,"act");

	if(in){ // first layer does not need weights or biases.
		init_matrix(&ret->weight,
		length, in->act.height,
		gen_one,ret, "weight");

		init_matrix(&ret->bias,
		length,1,
		gen_one,ret,"bias");

		init_matrix(&ret->weight_error,
		length,in->act.height,
		gen_zero,ret,"weight_error");

		init_matrix(&ret->bias_error,
		length,1,
		gen_zero,ret,"bias_error");
	}

	// link the list in both directions.
	ret->in = in;
	if(in)
		ret->in->out = ret;

	return ret;
}


struct net *init_network(
const int *spec, // list of number of neurons starting with input.
const int len) // depth of network.
{
	struct net *ret = malloc(sizeof(*ret));
	struct layer *tmp = init_layer(spec[0],"input",NULL);
	for(int i = 1; i < len; ++i)
		tmp = init_layer(spec[i],"other",tmp);

	tmp->out = NULL; // terminate linked list

	ret->top = tmp;
	ret->name = "Uaun";
	return ret;
}
void calc_net(struct net *const n)
{
	struct layer *layer_p = n->top;
	struct matrix tmp;
	init_matrix(&tmp,2,1,gen_zero,calc_tmp,"tmp");
	while(layer_p->in)
		layer_p = layer_p->in;
	layer_p = layer_p->out;
	ambpc(
	&layer_p->weight,
	&layer_p->in->act,
	&layer_p->z,
	&tmp);

}

const struct layer *get_input_layer(const struct layer *l)
{
	const struct layer *ret = l;
	while(ret->in)
		ret = ret->in;
	return ret;
}

void print_net_out(const struct net *const n)
{
	print_fmatrix(&n->top->act);
}
int main(void)
{
	srand(34);
	int spec[]={4,2};
	struct net *n = init_network(spec,2);

	print_net(n);
	calc_net(n);
	print_net(n);



	////calc z
	//ambpc(&weight_h,&act_in,&bias_h,&act_h);

	////peal off z
	//copy(&act_h,&act_h_z)

	////calc activation
	//mf_apply(&act_h,sigmoid);

	////calc z
	//ambpc(&weight_o,&act_h,&bias_o,&act_o);

	////peal off z
	//copy(&act_o,&act_o_tmp)

	////calc_activation
	//mf_apply(&act_o,sigmoid);

	//print_fmatrix(&act_o);

	//struct matrix error = {
	//	.name= "error",
	//	.t = CblasNoTrans,
	//	0};
	//copy(&act_o,&error);

	////backprop
	//first_error(&act_o,&act_o_tmp,&error);
	//error(&weight_h,&error,&act_h_z,&

	return 0;
}
