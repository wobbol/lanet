CC = clang
CLIBS = -l cblas -lm
DEBUG = -ggdb

all: gemm_test main


gemm_test:
	$(CC) gemm_test.c $(CLIBS) $(DEBUG) -o tm

main:
	$(CC) main.c $(CLIBS) $(DEBUG) -o ln
