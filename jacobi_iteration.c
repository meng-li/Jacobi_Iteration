/**
 * Description: Jacobi iteration to solve Ax=b of linear equations.
 * Author: lemon.li.hust@gmail.com
 */
#include <stdio.h>
#include <mpi.h>
#define MAX_ITERATIONS 100
#define ROOT 0

void init_data(int my_rank, double ***matrix_a, double **input_a, double **input_b,
        int *no_rows, int *no_cols);
void verify(int no_rows, int no_cols, int no_procs, int my_rank);
void iteration(double *x_new, double *x_old, double *x_bloc, double *a_recv,
        double *b_recv, int no_rows_bloc, int no_rows, int no_cols, int my_rank);
double cal_dist(double *x, double *y, int dim);
void display(int no_rows, int no_cols, double **matrix_a, double *input_b, 
        double *x, int my_rank); 

int main(int argc, char* argv[])
{
    int no_rows, no_cols, no_rows_bloc;
    int no_procs, my_rank;
    double **matrix_a, *input_a, *input_b, *a_recv, *b_recv;
    double *x_new, *x_old, *x_bloc;

    // MPI initialization
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &no_procs);

    // data initialization
    init_data(my_rank, &matrix_a, &input_a, &input_b, &no_rows, &no_cols);
    MPI_Bcast(&no_rows, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&no_cols, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    verify(no_rows, no_cols, no_procs, my_rank); 

    // iteration
    no_rows_bloc = no_rows / no_procs;
    x_new = (double *)malloc(no_rows * sizeof(double));
    x_old = (double *)malloc(no_rows * sizeof(double));
    x_bloc = (double *)malloc(no_rows_bloc * sizeof(double));
    a_recv = (double *)malloc(no_rows_bloc * no_cols * sizeof(double));
    b_recv = (double *)malloc(no_rows_bloc * no_cols * sizeof(double));
    MPI_Scatter(input_a, no_rows_bloc * no_cols, MPI_DOUBLE, a_recv, 
            no_rows_bloc * no_cols, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    MPI_Scatter(input_b, no_rows_bloc, MPI_DOUBLE, b_recv, no_rows_bloc,
            MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    iteration(x_new, x_old, x_bloc, a_recv, b_recv, no_rows_bloc, 
            no_rows, no_cols, my_rank);
    
    // display
    if(my_rank == ROOT)
        display(no_rows, no_cols, matrix_a, input_b, x_new, my_rank);
    
    MPI_Finalize();
    return 0;
}

void init_data(int my_rank, double ***matrix_a, double **input_a, double **input_b,
        int *no_rows, int *no_cols)
{
    int irow, icol, idx;
    FILE *fp;
    double **matrix_a_tmp, *input_a_tmp, *input_b_tmp;
    if(my_rank == ROOT) {
        // read matrix A
        if((fp=fopen("./matrix-data-jacobi.inp", "r")) == NULL) {
            printf("Can't open input file A \n");
            exit(-1);
        }
        fscanf(fp, "%d %d", no_rows, no_cols);
        matrix_a_tmp = (double **)malloc((*no_rows) * sizeof(double *));
        for(irow=0; irow<*no_rows; irow++) {
            matrix_a_tmp[irow] = (double *)malloc((*no_cols) * sizeof(double));
            for(icol=0; icol<*no_cols; icol++) {
                fscanf(fp, "%lf", &matrix_a_tmp[irow][icol]);
            }
        }
        fclose(fp);
        *matrix_a = matrix_a_tmp;
        
        // read vector B
        if((fp=fopen("./vector-data-jacobi.inp", "r")) == NULL) {
            printf("Can't open input vector file B");
            exit(-1);
        }
        fscanf(fp, "%d", no_rows);
        input_b_tmp = (double *)malloc((*no_rows) * sizeof(double));
        for(irow=0; irow<*no_rows; irow++) {
            fscanf(fp, "%lf", &input_b_tmp[irow]);
        }
        fclose(fp);
        *input_b = input_b_tmp;

        // convert matrix A into vector input A
        input_a_tmp = (double *)malloc((*no_rows) * (*no_cols) * sizeof(double));
        idx = 0;
        for(irow=0; irow<*no_rows; irow++) {
            for(icol=0; icol<*no_cols; icol++) {
                input_a_tmp[idx++] = matrix_a_tmp[irow][icol];
            }
        }
        *input_a = input_a_tmp;
    }
}

void verify(int no_rows, int no_cols, int no_procs, int my_rank)
{
    if(no_rows != no_cols) {
        MPI_Finalize();
        if(my_rank == ROOT) {
            printf("Matrix A should be square ...\n");
        }
        exit(-1);
    }
    if(no_rows % no_procs != 0) {
        MPI_Finalize();
        if(my_rank == ROOT) {
            printf("Matrix A can't be stripped even ...\n");
        }
        exit(-1);
    }
}

void iteration(double *x_new, double *x_old, double *x_bloc, double *a_recv,
        double *b_recv, int no_rows_bloc, int no_rows, int no_cols, int my_rank)
{
    int irow, icol, global_rows_idx, iter = 0;
    // initialize X[i] = B[i]
    for(irow=0; irow<no_rows_bloc; irow++) {
        x_bloc[irow] = b_recv[irow];
    }
    MPI_Allgather(x_bloc, no_rows_bloc, MPI_DOUBLE, x_new, no_rows_bloc, 
            MPI_DOUBLE, MPI_COMM_WORLD);
    do{
        for(irow=0; irow<no_rows; irow++) {
            x_old[irow] = x_new[irow];
        }
        for(irow=0; irow<no_rows_bloc; irow++) {
            global_rows_idx = (my_rank * no_rows_bloc) + irow;
            x_bloc[irow] = b_recv[irow];
            for(icol=0; icol<no_cols; icol++) {
                if(icol == global_rows_idx)
                    continue;
                x_bloc[irow] -= x_old[icol] * a_recv[irow * no_cols + icol];
            }
            x_bloc[irow] /= a_recv[irow * no_cols + global_rows_idx];
        }
        MPI_Allgather(x_bloc, no_rows_bloc, MPI_DOUBLE, x_new, no_rows_bloc, 
                MPI_DOUBLE, MPI_COMM_WORLD);
        iter++;
    } while((iter<MAX_ITERATIONS) && (cal_dist(x_old, x_new, no_rows)>=1.0E-24));
}

double cal_dist(double *x, double *y, int dim)
{
    int idx;
    double sum = 0.0;
    for(idx=0; idx<dim; idx++) {
        sum += (x[idx] - y[idx]) * (x[idx] - y[idx]);
    }
    return sum;
}

void display(int no_rows, int no_cols, double **matrix_a, double *input_b, 
        double *x, int my_rank) 
{
    int irow, icol;
    printf("\n");
    printf(" ------------------------------------------- \n");
    printf("Results of Jacobi Method on processor %d: \n", my_rank);
    printf("Matrix A: \n");
    for(irow=0; irow<no_rows; irow++) {
        for(icol=0; icol<no_cols; icol++) {
            printf("%.3lf ", matrix_a[irow][icol]);
        }
        printf("\n");
    }
    printf("\n");
    printf("Vector B: \n");
    for(irow=0; irow<no_rows; irow++) {
        printf("%.2lf \n", input_b[irow]);
    }
    printf("\n");
    printf("Solution vector: \n");
    for(irow=0; irow<no_rows; irow++) {
        printf("%.2lf\n", x[irow]);
    }
    printf(" --------------------------------------------------- \n");
}

