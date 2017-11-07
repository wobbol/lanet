#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

struct mat;

void print_mat(struct mat *m);
void mm(struct mat *a, struct mat *b, struct mat *out);
int max(const int a, const int b);
struct mat *init_mat(int h, int w, float *d);

struct mat{
	int h;
	int w;
	float d[];
};

int max(const int a, const int b)
{
	return (a < b) ? b:a;
}

struct mat *init_mat(int h, int w, float *d)
{
	struct mat *ret = malloc(sizeof(*ret)+(sizeof(ret->d[0])*h*w));

	ret->h = h;
	ret->w = w;

	for(int i = 0; i < (h*w); ++i)
		ret->d[i] = d[i];

	return ret;
}

void print_mat(struct mat *m)
{
	for(int j = 0; j < m->h; ++j){
		for(int i = 0; i < m->w; ++i)
			printf("%f ",m->d[i+j*m->w]);
		printf("\n");
	}
	return;
}

void mm(struct mat *a, struct mat *b, struct mat *out)
{// a*b=c
	assert(a->w == b->h);
	assert(a->h == out->h);
	assert(b->w == out->w);

	int common = a->w;

	for(int row_a = 0; row_a < a->h; ++row_a){
		for(int col_b = 0;col_b < b->w; ++col_b){
			float tmp = 0;
			for(int com = 0; com < common; ++com){
				int ida = com + row_a*a->w;
				int idb = com*b->w + col_b;
				assert(ida < (a->w*a->h));
				assert(idb < (b->w*b->h));

				tmp += (a->d[ida]) * (b->d[idb]);
			}
			int idc = col_b + row_a*out->w;
			assert(idc < (out->w*out->h));

			out->d[idc] = tmp;
		}
	}
	return;
}

int main(void)
{
	float a[] = {
		0.252015, 0.507907, 0.362560, 0.851875,
		0.294506, 0.475974, 0.612547, 0.227675};
	struct mat *A = init_mat(2,4,a);

	float b[] = {
		0.168970,
		0.028588,
		0.772811,
		0.364650};
	struct mat *B = init_mat(4,1,b);

	float c[] = {
		0,0,0,
		0,0,0};
	struct mat *C = init_mat(2,1,c);

	mm(A,B,C);
	print_mat(C);
	return 0;
}
