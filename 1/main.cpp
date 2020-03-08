#include <cstdio>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <complex>
#include <omp.h>

using namespace std;

typedef complex<double> complexd;

void
one_qubit_conversion(complexd *A, complexd *B, double H[4], int n, int k){
	long long int shift_n, shift_nk, inv_nk, change_bit = 0;
	shift_n = 1 << n;
	shift_nk = 1 << (n - k);
	inv_nk = ~shift_nk;
#pragma omp parallel for firstprivate(change_bit)
	for (long long int i = 0; i < shift_n; i++) {
		change_bit = (i & shift_nk) >> (n - k);
		B[i] = A[i & inv_nk] * H[change_bit * 2] + A[i | shift_nk] * H[change_bit * 2 + 1];
	}
}

int main(int argc, char **argv) {
	static const int ARG_CNT = 4;
	if (argc < ARG_CNT) {
		cerr << "Invalid call. Please type n, k, thr_num, test (0 or 1)." << endl;
		exit(1);
	}
	int n, k, thr_num, test;
	n = atoi(argv[1]);
	k = atoi(argv[2]);
	if (k > n) {
		cerr << "Invalid values." << endl;
		exit(1);
	}
	thr_num = atoi(argv[3]);
	omp_set_num_threads(thr_num);
	test = atoi(argv[4]);

	// A initialization
	long long int shift_n = 1 << n; // 2^n
	complexd *A = new complexd[shift_n];
	int real_part = 0, imag_part = 0;
	float half_max = RAND_MAX / 2;
	float sum = 0;
#pragma omp parallel for firstprivate(real_part, imag_part) reduction(+: sum)
	for (long long int i = 0; i < shift_n; i++) {
		srand(omp_get_wtime() + i);
		real_part = rand();
		if (real_part < half_max) {
			real_part *= -1;
		}
		real_part *= static_cast <float> (rand() / static_cast <float> (RAND_MAX));
		imag_part = rand();
		if (imag_part < half_max) {
			imag_part *= -1;
		}
		imag_part *= static_cast <float> (rand() / static_cast <float> (RAND_MAX));
		A[i].real(real_part);
		A[i].imag(imag_part);
		sum += abs(A[i] * A[i]);
	}
	sum = sqrt(sum);
#pragma omp parallel for
	for (long long int i = 0; i < shift_n; i++) {
		A[i] /= sum;
	}

	// Conversion
	double tmp_sqrt = 1 / sqrt(2);
	double H[4] = {tmp_sqrt, tmp_sqrt, tmp_sqrt, (-1) * tmp_sqrt};
	complexd *B = new complexd[shift_n];

	double begin, end, comp_time;
	begin = omp_get_wtime();
	one_qubit_conversion(A, B, H, n, k);
	end = omp_get_wtime();
	comp_time = end - begin;

	// Test output
	if (test == 1) {
		ofstream testfile;
		testfile.open("test.txt", ofstream::out | ofstream::trunc);
		for (long long int i = 0; i < shift_n; i++) {
			testfile << A[i] << " ";
		}
		testfile << endl << endl;
		for (long long int i = 0; i < shift_n; i++) {
			testfile << B[i] << " ";
		}
		testfile.close();
	}

	ofstream filename;
	filename.open("a.txt", ofstream::out | ofstream::app);
	filename << "n: " << n << " k: " << k << " thr_num: " << thr_num << " time: " << comp_time << endl;
	filename.close();

	delete [] A;
	delete [] B;	
	return 0;
}
