// gcc -o test test.c -lopenblas -lpthread 
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cblas.h"

#define INCX 1
#define INCY 1

typedef struct {
    int total;
    int pass;
    int fail;
} TestStats;

TestStats stats;

void result(const char* test_name, int pass) {
    stats.total++;
    if (pass) {
        stats.pass++;
        printf("[ПРОЙДЕН] %s\n", test_name);
    } else {
        stats.fail++;
        printf("[ПРОВАЛЕН] %s\n", test_name);
    }
}

void test_sgemv() {
    float A[1], x[1], y[1];
    int pass = 1;
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 0, 0, 1.0f, A, 1, x, INCX, 0.0f, y, INCY);
    result("sgemv интерфейс", pass);
}

void test_dgemv() {
    double A[1], x[1], y[1];
    int pass = 1;
    cblas_dgemv(CblasRowMajor, CblasNoTrans, 0, 0, 1.0, A, 1, x, INCX, 0.0, y, INCY);
    result("dgemv интерфейс", pass);
}

void test_cgemv() {
    float A[2], x[2], y[2];
    int pass = 1;
    float alpha[2] = {1.0f, 0.0f};
    float beta[2] = {0.0f, 0.0f};
    cblas_cgemv(CblasRowMajor, CblasNoTrans, 0, 0, alpha, A, 1, x, INCX, beta, y, INCY);
    result("cgemv интерфейс", pass);
}

void test_zgemv() {
    double A[2], x[2], y[2];
    int pass = 1;
    double alpha[2] = {1.0, 0.0};
    double beta[2] = {0.0, 0.0};
    cblas_zgemv(CblasRowMajor, CblasNoTrans, 0, 0, alpha, A, 1, x, INCX, beta, y, INCY);
    result("zgemv интерфейс", pass);
}

void test_ssymv() {
    float A[1], x[1], y[1];
    int pass = 1;
    cblas_ssymv(CblasRowMajor, CblasUpper, 0, 1.0f, A, 1, x, INCX, 0.0f, y, INCY);
    result("ssymv интерфейс", pass);
}

void test_dsymv() {
    double A[1], x[1], y[1];
    int pass = 1;
    cblas_dsymv(CblasRowMajor, CblasLower, 0, 1.0, A, 1, x, INCX, 0.0, y, INCY);
    result("dsymv интерфейс", pass);
}

void test_chemv() {
    float A[2], x[2], y[2];
    int pass = 1;
    float alpha[2] = {1.0f, 0.0f};
    float beta[2] = {0.0f, 0.0f};
    cblas_chemv(CblasRowMajor, CblasUpper, 0, alpha, A, 1, x, INCX, beta, y, INCY);
    result("chemv интерфейс", pass);
}

void test_zhemv() {
    double A[2], x[2], y[2];
    int pass = 1;
    double alpha[2] = {1.0, 0.0};
    double beta[2] = {0.0, 0.0};
    cblas_zhemv(CblasRowMajor, CblasUpper, 0, alpha, A, 1, x, INCX, beta, y, INCY);
    result("zhemv интерфейс", pass);
}

void test_strmv() {
    float A[1], x[1];
    int pass = 1;
    cblas_strmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    result("strmv интерфейс", pass);
}

void test_dtrmv() {
    double A[1], x[1];
    int pass = 1;
    cblas_dtrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    result("dtrmv интерфейс", pass);
}

void test_ctrmv() {
    float A[2], x[2];
    int pass = 1;
    cblas_ctrmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    result("ctrmv интерфейс", pass);
}

void test_ztrmv() {
    double A[2], x[2];
    int pass = 1;
    cblas_ztrmv(CblasRowMajor, CblasLower, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    result("ztrmv интерфейс", pass);
}

void test_strsv() {
    float A[1], x[1];
    int pass = 1;
    cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    result("strsv интерфейс", pass);
}

void test_dtrsv() {
    double A[1], x[1];
    int pass = 1;
    cblas_dtrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit, 0, A, 1, x, INCX);
    result("dtrsv интерфейс", pass);
}

void test_ctrsv() {
    float A[2], x[2];
    int pass = 1;
    cblas_ctrsv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 0, A, 1, x, INCX);
    result("ctrsv интерфейс", pass);
}

void test_ztrsv() {
    double A[2], x[2];
    int pass = 1;
    cblas_ztrsv(CblasRowMajor, CblasLower, CblasNoTrans, CblasUnit, 0, A, 1, x, INCX);
    result("ztrsv интерфейс", pass);
}

void test_sger() {
    float A[1], x[1], y[1];
    int pass = 1;
    cblas_sger(CblasRowMajor, 0, 0, 1.0f, x, INCX, y, INCY, A, 1);
    result("sger интерфейс", pass);
}

void test_dger() {
    double A[1], x[1], y[1];
    int pass = 1;
    cblas_dger(CblasRowMajor, 0, 0, 1.0, x, INCX, y, INCY, A, 1);
    result("dger интерфейс", pass);
}

void test_cgeru() {
    float A[2], x[2], y[2];
    int pass = 1;
    float alpha[2] = {1.0f, 0.0f};
    cblas_cgeru(CblasRowMajor, 0, 0, alpha, x, INCX, y, INCY, A, 1);
    result("cgeru интерфейс", pass);
}

void test_zgeru() {
    double A[2], x[2], y[2];
    int pass = 1;
    double alpha[2] = {1.0, 0.0};
    cblas_zgeru(CblasRowMajor, 0, 0, alpha, x, INCX, y, INCY, A, 1);
    result("zgeru интерфейс", pass);
}

void test_cgerc() {
    float A[2], x[2], y[2];
    int pass = 1;
    float alpha[2] = {1.0f, 0.0f};
    cblas_cgerc(CblasRowMajor, 0, 0, alpha, x, INCX, y, INCY, A, 1);
    result("cgerc интерфейс", pass);
}

void test_zgerc() {
    double A[2], x[2], y[2];
    int pass = 1;
    double alpha[2] = {1.0, 0.0};
    cblas_zgerc(CblasRowMajor, 0, 0, alpha, x, INCX, y, INCY, A, 1);
    result("zgerc интерфейс", pass);
}

void test_ssyr() {
    float A[1], x[1];
    int pass = 1;
    cblas_ssyr(CblasRowMajor, CblasLower, 0, 1.0f, x, INCX, A, 1);
    result("ssyr интерфейс", pass);
}

void test_dsyr() {
    double A[1], x[1];
    int pass = 1;
    cblas_dsyr(CblasRowMajor, CblasUpper, 0, 1.0, x, INCX, A, 1);
    result("dsyr интерфейс", pass);
}

void test_cher() {
    float A[2], x[2];
    int pass = 1;
    float alpha = 1.0f;
    cblas_cher(CblasRowMajor, CblasUpper, 0, alpha, x, INCX, A, 1);
    result("cher интерфейс", pass);
}

void test_zher() {
    double A[2], x[2];
    int pass = 1;
    double alpha = 1.0;
    cblas_zher(CblasRowMajor, CblasUpper, 0, alpha, x, INCX, A, 1);
    result("zher интерфейс", pass);
}

void test_ssyr2() {
    float A[1], x[1], y[1];
    int pass = 1;
    cblas_ssyr2(CblasRowMajor, CblasLower, 0, 1.0f, x, INCX, y, INCY, A, 1);
    result("ssyr2 интерфейс", pass);
}

void test_dsyr2() {
    double A[1], x[1], y[1];
    int pass = 1;
    cblas_dsyr2(CblasRowMajor, CblasUpper, 0, 1.0, x, INCX, y, INCY, A, 1);
    result("dsyr2 интерфейс", pass);
}

void test_cher2() {
    float A[2], x[2], y[2];
    int pass = 1;
    float alpha[2] = {1.0f, 0.0f};
    cblas_cher2(CblasRowMajor, CblasUpper, 0, alpha, x, INCX, y, INCY, A, 1);
    result("cher2 интерфейс", pass);
}

void test_zher2_failing() {
    double A[1] = {1.0};
    double x[1] = {1.0};
    double y[1] = {1.0};
    double alpha[2] = {1.0, 0.0};
    cblas_zher2(CblasRowMajor, CblasUpper, 0, alpha, x, INCX, y, INCY, A, 1);
    result("zher2 интерфейс", 0);
}

void test_sgemv_failing() {
    float A[4] = {1,2,3,4};
    float x[2] = {1,1};
    float y[2] = {0,0};
    cblas_sgemv(CblasRowMajor, CblasNoTrans, 2, 2, 1.0f, A, 1, x, INCX, 0.0f, y, INCY);
    result("sgemv lda error", 0);
}

void test_ssymv_failing() {
    float A[4] = {1,2,3,4};
    float x[3] = {1,1,1};
    float y[3] = {0,0,0};
    cblas_ssymv(CblasRowMajor, CblasUpper, 3, 1.0f, A, 2, x, INCX, 0.0f, y, INCY);
    result("ssymv buffer error", 0);
}

void test_strmv_failing() {
    float A[4] = {1,2,3,4};
    float x[3] = {1,1,1};
    cblas_strmv(CblasRowMajor, CblasUpper, CblasNoTrans, CblasNonUnit, 3, A, 2, x, INCX);
    result("strmv lda error", 0);
}

void test_dger_failing() {
    double A[4] = {1,2,3,4};
    double x[3] = {1,1,1};
    double y[2] = {1,1};
    cblas_dger(CblasRowMajor, 2, 2, 1.0, x, 1, y, 1, A, 2);
    result("dger dimension error", 0);
}

int main() {
    printf("CBLAS Level 2 Интерфейсные Тесты\n");
    printf("===================================\n\n");
    
    stats.total = 0;
    stats.pass = 0;
    stats.fail = 0;
    
    test_sgemv();
    test_dgemv();
    test_cgemv();
    test_zgemv();
    
    test_ssymv();
    test_dsymv();
    test_chemv();
    test_zhemv();
    
    test_strmv();
    test_dtrmv();
    test_ctrmv();
    test_ztrmv();
    
    test_strsv();
    test_dtrsv();
    test_ctrsv();
    test_ztrsv();
    
    test_sger();
    test_dger();
    test_cgeru();
    test_zgeru();
    test_cgerc();
    test_zgerc();
    
    test_ssyr();
    test_dsyr();
    test_cher();
    test_zher();
    
    test_ssyr2();
    test_dsyr2();
    test_cher2();
    test_zher2_failing();
    test_sgemv_failing();
    test_ssymv_failing();
    test_strmv_failing();
    test_dger_failing();
    
    printf("\n===================================\n");
    printf("Всего тестов: %d\n", stats.total);
    printf("Пройдено: %d\n", stats.pass);
    printf("Провалено: %d\n", stats.fail);
    printf("===================================\n");
    
    return (stats.fail == 0) ? 0 : 1;
}