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
} thread;

void* multiply(void* arg) {
    thread* d = (thread*)arg;
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

void my_sgemm(float* A, float* B, float* C, int n, int threads) {
    pthread_t t[16];           
    thread data[16];
    
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
        
        pthread_create(&t[i], NULL, multiply, &data[i]);
    }
    
    for (int i = 0; i < threads; i++) {
        pthread_join(t[i], NULL);
    }
}

void init_matrix(float* m, int n) {
    for (int i = 0; i < n * n; i++) {
        m[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    double start_time = get_time();
    
    int n = 1000;
    int threads[] = {1, 2, 4, 8, 16};
    int repeats = 5;
    
    float *A = malloc(n * n * sizeof(float));
    float *B = malloc(n * n * sizeof(float));
    float *C1 = malloc(n * n * sizeof(float));
    float *C2 = malloc(n * n * sizeof(float));
    
    srand(time(NULL));
    init_matrix(A, n);
    init_matrix(B, n);
    
    printf("Потоки |  Наш код   | BLAS время | Производит | Ср.геом\n");
    printf("----------------------------------------------------\n");
    
    for (int t = 0; t < 5; t++) {
        int th = threads[t];
        
        double my_sum = 0, blas_sum = 0;
        double my_t[5], blas_t[5];
        
        for (int r = 0; r < repeats; r++) {
            for (int i = 0; i < n * n; i++) C1[i] = 0;
            
            double a = get_time();
            my_sgemm(A, B, C1, n, th);
            double b = get_time();
            my_t[r] = b - a;
            my_sum += my_t[r];
  
            a = get_time();
            openblas_set_num_threads(th);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
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
        
        printf("%3d    |  %.4f    |  %.4f    |   %.1f%%     | %.1f%%\n", 
               th, my_avg, blas_avg, perf, gmean);
    }
    
    free(A);
    free(B);
    free(C1);
    free(C2);
    
    double end_time = get_time();
    int total = (int)(end_time - start_time);
    
    printf("\nОбщее время программы: %d секунд\n", total);
    
    return 0;
}