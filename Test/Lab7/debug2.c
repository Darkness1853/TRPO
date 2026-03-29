#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


void *work(int, int, int **);
void *inc(int **, int );
void *init(int ***, int);
void *destroy(int **, int);
long long int  m_sum(int **, int);

static unsigned long long N = 10000;

/***************************************************************/
int main(int argc, char **argv) {

    int i, num_ths;
	long long int res, t_res;
	
	printf("program start\n");

    if (argc != 2) {
        printf("usage: ./a.out num_threads\n");
        exit(1);
    }
    num_ths = atoi(argv[1]);
    
    if ((num_ths < 1) || (num_ths > 24)) { num_ths = 6; }
	omp_set_num_threads(num_ths);
	#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("thread %d: Hello\n", tid);
    }
	
	int **matrix = NULL;
    init(&matrix, N);
	inc(matrix, N);
	res = m_sum(matrix, N);
	t_res = 2*((long long int)N)*((long long int)N);
	printf("program's sum of elements of matrix  = %lld\n", res);
	printf("theoretical(right) sum of elements of matrix  = %lld\n", t_res);
	destroy(matrix, N);
	exit(0);
}


void *init(int ***m, int n){
	
	*m = (int **)malloc(sizeof(int *)*n);
	if (!(*m)) {
        perror("malloc matrix array");
        exit(1);
    }
	
	for (int i = 0; i < n; i++) {
		if (!( (*m)[i] = (int *)malloc(sizeof(int)*n))) {
        perror("malloc matrix array row");
        exit(1);
		}
	}
	
	for (int i = 0; i < n; i++){
		for(int j = 0; j < n; j++)
			(*m)[i][j] = 1;
	}
	return (void *)0;
}

void *destroy(int **m, int n){
	
	for (int i = 0; i < n; i++) {
		free(m[i]);
		}
	free(m);
}

void *work(int i, int j,int **m) {
	m[i][j]++;
	return (void *)0;
}

void *inc(int **m, int n){
	int i,j;
	#pragma omp parallel private(i,j)
	{
       #pragma omp for 
       for (i=0; i < n; i++)
               for (j=0; j < n; j++)
				   work(i, j, m);
    }
	return (void *)0;
}

long long int  m_sum(int **m, int n){
	long long int res = 0; 
	#pragma omp parallel for reduction(+:res) shared(m) collapse(2)
       for (int i=0; i < n; i++){
		   for (int j=0; j < n; j++)
			   res+=m[i][j];
   }
   return res;
}
