all:
	g++ -O3 -Wall -Werror -fopenmp main.cpp -o main
polus: all
	for n in 20 24 28 30; do \
		for k in 1 5 $$n; do \
			for thr_num in 1 2 4 8 16 32 64 128; do \
				bsub -W 15:00 -q normal ./main $$n $$k $$thr_num; \
			done \
		done \
	done
