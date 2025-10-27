CC = gcc
CFLAGS = -fopenmp -O3 -Wall -std=c99

all: nqueen_arts_local nqueen_arts_random nqueen_arts_explicit
# nqueen_seq nqueen_omp nqueen_omp_opt nqueen_arts_seq nqueen_carts

# nqueen_seq: nqueen_seq.c
# 	$(CC) $(CFLAGS) -o $@ $<

# nqueen_omp: nqueen_omp.c
# 	$(CC) $(CFLAGS) -o $@ $<

# nqueen_omp_opt: nqueen_omp_opt.c
# 	$(CC) $(CFLAGS) -o $@ $<

# nqueen_arts_seq: nqueen_arts_seq.c
# 	$(CC) $(CFLAGS) -o $@ $< -I../../.install/include -L../../.install/lib -larts

# nqueen_carts: nqueen_omp.c
# 	carts execute $< -o $@

nqueen_arts_local: nqueen_arts_local.c
	$(CC) $(CFLAGS) -o $@ $< -I../../.install/include -L../../.install/lib -L../../.install/lib64 -larts -lhwloc

nqueen_arts_random: nqueen_arts_random.c
	$(CC) $(CFLAGS) -o $@ $< -I../../.install/include -L../../.install/lib -L../../.install/lib64 -larts -lhwloc

nqueen_arts_explicit: nqueen_arts_explicit.c
	$(CC) $(CFLAGS) -o $@ $< -I../../.install/include -L../../.install/lib -L../../.install/lib64 -larts -lhwloc

clean:
	rm -f nqueen_arts_local nqueen_arts_random nqueen_arts_explicit
 
.PHONY: all clean
