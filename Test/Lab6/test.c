#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <pthread.h>
#include "../OpenBLAS/cblas.h"

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

typedef struct {
    float* A;
    float* B;
    float* C;
    int n;
    int start;
    int end;
} thread_float;

void* multiply_float(void* arg) {
    thread_float* d = (thread_float*)arg;
    int n = d->n;
    
    for (int i = d->start; i < d->end; i++) {
        float* c_row = &d->C[i * n];
        for (int k = 0; k < n; k++) {
            float aik = d->A[i * n + k];
            float* b_row = &d->B[k * n];
            for (int j = 0; j < n; j++) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
    return NULL;
}

void my_sgemm_float(float* A, float* B, float* C, int n, int threads) {
    pthread_t t[16];           
    thread_float data[16];
    
    int rows = n / threads;
    int rem = n % threads;
    int cur = 0;
    
    for (int i = 0; i < threads; i++) {
        data[i].A = A;
        data[i].B = B;
        data[i].C = C;
        data[i].n = n;
        data[i].start = cur;
        
        int extra = (i < rem) ? 1 : 0;
        data[i].end = cur + rows + extra;
        cur = data[i].end;
        
        pthread_create(&t[i], NULL, multiply_float, &data[i]);
    }
    
    for (int i = 0; i < threads; i++) {
        pthread_join(t[i], NULL);
    }
}

typedef struct {
    double* A;
    double* B;
    double* C;
    int n;
    int start;
    int end;
} thread_double;

void* multiply_double(void* arg) {
    thread_double* d = (thread_double*)arg;
    int n = d->n;
    
    for (int i = d->start; i < d->end; i++) {
        double* c_row = &d->C[i * n];
        for (int k = 0; k < n; k++) {
            double aik = d->A[i * n + k];
            double* b_row = &d->B[k * n];
            for (int j = 0; j < n; j++) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
    return NULL;
}

void my_sgemm_double(double* A, double* B, double* C, int n, int threads) {
    pthread_t t[16];           
    thread_double data[16];
    
    int rows = n / threads;
    int rem = n % threads;
    int cur = 0;
    
    for (int i = 0; i < threads; i++) {
        data[i].A = A;
        data[i].B = B;
        data[i].C = C;
        data[i].n = n;
        data[i].start = cur;
        
        int extra = (i < rem) ? 1 : 0;
        data[i].end = cur + rows + extra;
        cur = data[i].end;
        
        pthread_create(&t[i], NULL, multiply_double, &data[i]);
    }
    
    for (int i = 0; i < threads; i++) {
        pthread_join(t[i], NULL);
    }
}

void init_matrix_float(float* m, int n) {
    for (int i = 0; i < n * n; i++) {
        m[i] = (float)rand() / RAND_MAX;
    }
}

void init_matrix_double(double* m, int n) {
    for (int i = 0; i < n * n; i++) {
        m[i] = (double)rand() / RAND_MAX;
    }
}

void run_float_test(int n, int threads[], int repeats) {
    float *A = malloc(n * n * sizeof(float));
    float *B = malloc(n * n * sizeof(float));
    float *C1 = malloc(n * n * sizeof(float));
    float *C2 = malloc(n * n * sizeof(float));
    
    init_matrix_float(A, n);
    init_matrix_float(B, n);
    
    printf("\nТЕСТ FLOAT\n");
    printf("Потоки |  Наш код   | BLAS время | Производит | Ср.геом  | BLAS быстрее в\n");
    printf("--------------------------------------------------------------------------\n");
    
    for (int t = 0; t < 5; t++) {
        int th = threads[t];
        
        double my_sum = 0, blas_sum = 0;
        double my_t[5], blas_t[5];
        
        for (int r = 0; r < repeats; r++) {
            for (int i = 0; i < n * n; i++) C1[i] = 0;
            
            double a = get_time();
            my_sgemm_float(A, B, C1, n, th);
            double b = get_time();
            my_t[r] = b - a;
            my_sum += my_t[r];
  
            a = get_time();
            openblas_set_num_threads(th);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       n, n, n, 1.0f, A, n, B, n, 0.0f, C2, n);
            b = get_time();
            blas_t[r] = b - a;
            blas_sum += blas_t[r];
        }
        
        double my_avg = my_sum / repeats;
        double blas_avg = blas_sum / repeats;
        double perf = (blas_avg / my_avg) * 100;
        
        double gsum = 0;
        for (int r = 0; r < repeats; r++) {
            gsum += log(blas_t[r] / my_t[r]);
        }
        double gmean = exp(gsum / repeats) * 100;

        double crat_speed = my_avg / blas_avg;
        
        printf("%3d    |  %.4f    |  %.4f    |   %.1f%%     | %.1f%%     | %.2f\n", 
               th, my_avg, blas_avg, perf, gmean, crat_speed);
    }
    
    free(A);
    free(B);
    free(C1);
    free(C2);
}

void run_double_test(int n, int threads[], int repeats) {
    double *A = malloc(n * n * sizeof(double));
    double *B = malloc(n * n * sizeof(double));
    double *C1 = malloc(n * n * sizeof(double));
    double *C2 = malloc(n * n * sizeof(double));
    
    init_matrix_double(A, n);
    init_matrix_double(B, n);
    
    printf("\nТЕСТ DOUBLE\n");
    printf("Потоки |  Наш код   | BLAS время | Производит | Ср.геом | BLAS быстрее в\n");
    printf("--------------------------------------------------------------------------\n");
    
    for (int t = 0; t < 5; t++) {
        int th = threads[t];
        
        double my_sum = 0, blas_sum = 0;
        double my_t[5], blas_t[5];
        
        for (int r = 0; r < repeats; r++) {
            for (int i = 0; i < n * n; i++) C1[i] = 0.0;
            
            double a = get_time();
            my_sgemm_double(A, B, C1, n, th);
            double b = get_time();
            my_t[r] = b - a;
            my_sum += my_t[r];
  
            a = get_time();
            openblas_set_num_threads(th);
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                       n, n, n, 1.0, A, n, B, n, 0.0, C2, n);
            b = get_time();
            blas_t[r] = b - a;
            blas_sum += blas_t[r];
        }
        
        double my_avg = my_sum / repeats;
        double blas_avg = blas_sum / repeats;
        double perf = (blas_avg / my_avg) * 100;
        
        double gsum = 0;
        for (int r = 0; r < repeats; r++) {
            gsum += log(blas_t[r] / my_t[r]);
        }
        double gmean = exp(gsum / repeats) * 100;

        double crat_speed = my_avg / blas_avg;
        
        printf("%3d    |  %.4f    |  %.4f    |   %.1f%%     | %.1f%%     | %.2f\n", 
               th, my_avg, blas_avg, perf, gmean, crat_speed);
    }
    
    free(A);
    free(B);
    free(C1);
    free(C2);
}

int main() {
    double start_time = get_time();
    
    int n = 1000;
    int threads[] = {1, 2, 4, 8, 16};
    int repeats = 5;
    
    srand(time(NULL));

    run_float_test(n, threads, repeats);

    run_double_test(n, threads, repeats);
    
    double end_time = get_time();
    int total = (int)(end_time - start_time);
    
    printf("\nОбщее время программы: %d секунд\n", total);
    
    return 0;
}