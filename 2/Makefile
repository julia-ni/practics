all:
	 mpic++ -O3 -Wall -Werror main.cpp -lm -o main
polus: all
	for n in 20 24 28 30; do \
		for k in 1 5 $$n; do \
			for thr_num in 1 2 4 8 16 32 64; do \
				bsub -W 15:00 -q normal ./main 0 $$n $$k a b; \
			done \
		done \
	done