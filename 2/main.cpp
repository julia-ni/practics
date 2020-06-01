#include <cstdio>
#include <cmath>
#include <ctime>
#include <iostream>
#include <fstream>
#include <complex>
#include "time.h"
#include <mpi.h>

using namespace std;

#define MASTER 0

typedef complex<double> complexd;

complexd *
vector_generation(int rank, int size, int n, char *file_name) {
	// Generation part
	unsigned long long shift_n = (1LLU << n) / size;
	complexd *A = new complexd[shift_n];
	double real_part = 0, imag_part = 0;
	double half_max = RAND_MAX / 2;
	double sum = 0, all_sum = 0;
	unsigned int seed;
	srand(MPI_Wtime());
	seed = rand() * rank;
	for (unsigned long long i = 0; i < shift_n; i++) {
		real_part = rand_r(&seed);
		if (real_part < half_max) {
			real_part *= -1;
		}
		real_part *= static_cast <double> (rand_r(&seed) / static_cast <double> (RAND_MAX));
		imag_part = rand_r(&seed);
		if (imag_part < half_max) {
			imag_part *= -1;
		}
		imag_part *= static_cast <double> (rand_r(&seed) / static_cast <double> (RAND_MAX));
		A[i].real(real_part);
		A[i].imag(imag_part);
		sum += abs(A[i] * A[i]);
	}
	MPI_Reduce(&sum, &all_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == MASTER) {
		all_sum = sqrt(all_sum);
	}
	MPI_Bcast(&all_sum, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for (unsigned long long i = 0; i < shift_n; i++) {
		A[i] /= all_sum;
	}

	// Output part
	MPI_File outf;
	if (MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outf)) {
		if (rank == MASTER) {
			cerr << "Cannot open the file." << endl;
		}
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	}
	MPI_File_seek(outf, shift_n * 2 * rank * sizeof(double), MPI_SEEK_SET);
	for (unsigned long long i = 0; i < shift_n; i++) {
		real_part = A[i].real();
		imag_part = A[i].imag();
		MPI_File_write(outf, &real_part, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
		MPI_File_write(outf, &imag_part, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
	}
	MPI_File_close(&outf);
	return A;
}

void one_qubit_conversion(int rank, int size, complexd *A, complexd *B, double H[4], int n, int k) {
	unsigned long long shift_n, shift_nk, inv_nk, change_bit = 0;
	shift_n = 1LLU << n;
	shift_nk = 1LLU << (n - k);
	inv_nk = ~shift_nk;

	int m = n * rank;
	int neighbour_rank = (m ^ (1u << (k - 1))) / n; 
	if (rank != neighbour_rank) {
		MPI_Sendrecv(A, n, MPI_DOUBLE_COMPLEX, neighbour_rank, 0, B, m, MPI_DOUBLE_COMPLEX, neighbour_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		if (rank < neighbour_rank) {
			for (unsigned long long i = 0; i < n; i++) {
				B[i] = A[i] * H[0] + B[i] * H[1];
			}
		} else {
			for (unsigned long long i = 0; i < n; i++) {
				B[i] = A[i] * H[3] + B[i] * H[2];
			}
		}
	} else {
		unsigned long long shift_nk = 1LLU << ((int)log2(n) - k);
		for (unsigned long long i = 0; i < n; i++) {
			change_bit = (i & shift_nk) >> (n - k);
			B[i] = A[i & inv_nk] * H[change_bit * 2] + A[i | shift_nk] * H[change_bit * 2 + 1];
		}
	}
}

int
main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (argc < 4) {
		if (rank == MASTER) {
			cerr << "Wrong number of parameters." << endl;
		}
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	}

	int test, n;
	char *file_name;
	test = atoi(argv[1]);
	n = atoi(argv[2]);
	int k = atoi(argv[3]);
	if (k > n) {
		if (rank == MASTER) {
			cerr << "Invalid values." << endl;
		}
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	}

	unsigned long long shift_n = (1LLU << n) / size;
	double real_part, imag_part;
	complexd *A = new complexd[shift_n];
	file_name = argv[4];
	// Test mode
	if (test == 1) {
		A = vector_generation(rank, size, n, argv[4]); // generating func
	} else {
		// Input
		MPI_File fin;
		if (MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_RDONLY, MPI_INFO_NULL, &fin)) {
			if (rank == MASTER) {
				cerr << "Cannot open the file." << endl;
			}
			MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
		}
		
		MPI_File_seek(fin, shift_n * 2 * rank * sizeof(double), MPI_SEEK_SET);
		for (unsigned long long i; i < shift_n; i++) {
			MPI_File_read(fin, &real_part, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
			MPI_File_read(fin, &imag_part, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
			A[i].real(real_part);
			A[i].imag(imag_part);
		}
		MPI_File_close(&fin);
	}


	// Conversion
	MPI_Barrier(MPI_COMM_WORLD);
	double tmp_sqrt = 1 / sqrt(2);
	double H[4] = {tmp_sqrt, tmp_sqrt, tmp_sqrt, (-1) * tmp_sqrt};
	complexd *B = new complexd[shift_n];
	
	double begin, end, time, all_time;
	begin = MPI_Wtime();
	one_qubit_conversion(rank, size, A, B, H, shift_n, k);
	end = MPI_Wtime();
	time = end - begin;
	MPI_Reduce(&time, &all_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	// Output
	file_name = argv[5];
	MPI_File outf;
	if (MPI_File_open(MPI_COMM_WORLD, file_name, MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_APPEND, MPI_INFO_NULL, &outf)) {
		if (rank == MASTER) {
			cerr << "Cannot open the file." << endl;
		}
		MPI_Abort(MPI_COMM_WORLD, MPI_ERR_OTHER);
	}
	MPI_File_seek(outf, 2 * shift_n * rank * sizeof(double), MPI_SEEK_SET);
	for (unsigned long long i = 0; i < shift_n; i++) {
		real_part = B[i].real();
		imag_part = B[i].imag();
		MPI_File_write(outf, &real_part, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
		MPI_File_write(outf, &imag_part, 1, MPI_DOUBLE, MPI_STATUS_IGNORE);
	}
	MPI_File_close(&outf);

	// Time results
	if (rank == MASTER) {
		ofstream fres;
		fres.open("res.txt", ofstream::out | ofstream::app);
		fres << "n: " << n << " k: " << k << " size: " << size << " time: " << all_time << endl;
		fres.close();
	}

	delete [] A;
	delete [] B;
	return MPI_Finalize();
}