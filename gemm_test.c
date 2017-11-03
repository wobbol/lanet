#include <cblas.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
int max(const int a, const int b)
{
	return (a < b) ? b:a;
}
struct mat{
	int h;
	int w;
	float d[];
};

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
				ida++;
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
	0.168970, 0.028588, 0.772811, 0.364650,
	0.237839, 0.487580, 0.252015, 0.507907};
	struct mat *A = init_mat(2,4,a);

	float b[] = {
		1,1,1,
		1,1,1,
		1,1,1,
		1,1,1};
	struct mat *B = init_mat(4,3,b);
	float c[] = {
		0,0,0,
		0,0,0};
	struct mat *C = init_mat(2,3,c);

	mm(A,B,C);

	print_mat(C);

	CBLAS_LAYOUT layout = CblasRowMajor;
	CBLAS_TRANSPOSE ta = CblasNoTrans;
	CBLAS_TRANSPOSE tb = CblasNoTrans;
	int rows_a_c = 2;
	int cols_b_c = 1;
	int col_a_row_b = 4;
	int lda, ldb, ldc;
	int alpha = 1;
	int beta = 1;
	//if(ta == CblasNoTrans){
	//	lda = max(1,a->height);
	//	ldb = max(1,b->height);
	//	ldc = max(1,c->height);
	//} else {
	//	lda = max(1,col_a_row_b);
	//	ldb = max(1,cols_b_c);
	//	ldc = max(1,rows_a_c);
	//}
	/*
	 * a <- m by k matrix
	 * b <- k by n matrix
	 * c <- m by n matrix
	 * ldx = row stride
	 * a*b+c->c
	 * 1,1 * 1,1 + 1,1
	 * 1,1   1,1   1,1
	 * 1,1         1,1
	 *
	 * */


	return 0;
}
