#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Function to check if placing a queen at (row, col) is safe
int isSafe(int board[], int row, int col) {
  for (int i = 0; i < row; i++) {
    // Check column and diagonal attacks
    if (board[i] == col || board[i] - i == col - row ||
        board[i] + i == col + row) {
      return 0;
    }
  }
  return 1;
}

// Sequential backtracking function
int solveSequential(int board[], int row, int n) {
  if (row == n) {
    return 1;
  }

  int count = 0;
  for (int col = 0; col < n; col++) {
    if (isSafe(board, row, col)) {
      board[row] = col;
      count += solveSequential(board, row + 1, n);
    }
  }
  return count;
}

// FIXED VERSION 1: Parallel for with dynamic scheduling
int solveParallelForDynamic(int n) {
  int total_solutions = 0;

#pragma omp parallel for reduction(+ : total_solutions) schedule(dynamic)
  for (int first_col = 0; first_col < n; first_col++) {
    int *board = (int *)malloc(n * sizeof(int));
    board[0] = first_col;

    int local_count = solveSequential(board, 1, n);
    total_solutions += local_count;

    free(board);
  }

  return total_solutions;
}

// FIXED VERSION 2: Task-based with proper granularity control
int solveTaskBasedFixed(int n) {
  int total_solutions = 0;

#pragma omp parallel shared(total_solutions)
  {
#pragma omp single
    {
      // Create tasks only for the first level to avoid overhead
      for (int first_col = 0; first_col < n; first_col++) {
#pragma omp task firstprivate(first_col, n) shared(total_solutions)
        {
          int *board = (int *)malloc(n * sizeof(int));
          board[0] = first_col;

          int local_count = solveSequential(board, 1, n);

#pragma omp atomic
          total_solutions += local_count;

          free(board);
        }
      }
    }
  }

  return total_solutions;
}

// FIXED VERSION 3: Memory-optimized with thread-local storage
int solveMemoryOptimized(int n) {
  int total_solutions = 0;

#pragma omp parallel reduction(+ : total_solutions)
  {
    // Each thread allocates its board once
    int *thread_board = (int *)malloc(n * sizeof(int));

#pragma omp for schedule(dynamic)
    for (int first_col = 0; first_col < n; first_col++) {
      thread_board[0] = first_col;
      int local_count = solveSequential(thread_board, 1, n);
      total_solutions += local_count;
    }

    free(thread_board);
  }

  return total_solutions;
}

// FIXED VERSION 4: Nested parallelism (if available)
int solveNestedParallel(int n) {
  int total_solutions = 0;

  // Enable nested parallelism if available
  omp_set_nested(1);

#pragma omp parallel for reduction(+ : total_solutions) schedule(dynamic)      \
    num_threads(4)
  for (int first_col = 0; first_col < n; first_col++) {
    int first_level_count = 0;

#pragma omp parallel for reduction(+ : first_level_count) schedule(dynamic)    \
    num_threads(5)
    for (int second_col = 0; second_col < n; second_col++) {
      if (second_col == first_col || abs(second_col - first_col) == 1) {
        continue; // Not safe
      }

      int *board = (int *)malloc(n * sizeof(int));
      board[0] = first_col;
      board[1] = second_col;

      int local_count = solveSequential(board, 2, n);
      first_level_count += local_count;

      free(board);
    }

    total_solutions += first_level_count;
  }

  return total_solutions;
}

// FIXED VERSION 5: Work-stealing with optimized chunk size
int solveWorkStealing(int n) {
  int total_solutions = 0;
  int num_threads = omp_get_max_threads();
  int chunk_size = (n + num_threads - 1) / num_threads;

  if (chunk_size < 1)
    chunk_size = 1;

#pragma omp parallel for reduction(+ : total_solutions)                        \
    schedule(guided, chunk_size)
  for (int first_col = 0; first_col < n; first_col++) {
    int *board = (int *)malloc(n * sizeof(int));
    board[0] = first_col;

    int local_count = solveSequential(board, 1, n);
    total_solutions += local_count;

    free(board);
  }

  return total_solutions;
}

int main(int argc, char *argv[]) {
  int n = 8;

  if (argc > 1) {
    n = atoi(argv[1]);
    if (n < 1 || n > 20) {
      printf("Board size must be between 1 and 20\n");
      return 1;
    }
  }

  printf("N-Queens Problem: Optimized Implementations Comparison\n");
  printf("Board size: %d\n", n);
  printf("Available threads: %d\n", omp_get_max_threads());
  printf("========================================================\n");

  double start_time, end_time;
  int solutions;

  // Method 1: Parallel for with dynamic scheduling
  printf("\n1. Parallel for (dynamic scheduling):\n");
  start_time = omp_get_wtime();
  solutions = solveParallelForDynamic(n);
  end_time = omp_get_wtime();
  printf("   Solutions: %d, Time: %.4f seconds\n", solutions,
         end_time - start_time);

  // Method 2: Fixed task-based approach
  printf("\n2. Task-based (fixed granularity):\n");
  start_time = omp_get_wtime();
  solutions = solveTaskBasedFixed(n);
  end_time = omp_get_wtime();
  printf("   Solutions: %d, Time: %.4f seconds\n", solutions,
         end_time - start_time);

  // Method 3: Memory-optimized
  printf("\n3. Memory-optimized (thread-local storage):\n");
  start_time = omp_get_wtime();
  solutions = solveMemoryOptimized(n);
  end_time = omp_get_wtime();
  printf("   Solutions: %d, Time: %.4f seconds\n", solutions,
         end_time - start_time);

  // Method 4: Work-stealing
  printf("\n4. Work-stealing (guided scheduling):\n");
  start_time = omp_get_wtime();
  solutions = solveWorkStealing(n);
  end_time = omp_get_wtime();
  printf("   Solutions: %d, Time: %.4f seconds\n", solutions,
         end_time - start_time);

  // Method 5: Nested parallelism (only for larger problems)
  if (n >= 10) {
    printf("\n5. Nested parallelism:\n");
    start_time = omp_get_wtime();
    solutions = solveNestedParallel(n);
    end_time = omp_get_wtime();
    printf("   Solutions: %d, Time: %.4f seconds\n", solutions,
           end_time - start_time);
  }

  // Verification
  int expected[] = {0,  1,   0,   0,    2,     10,    4,     40,
                    92, 352, 724, 2680, 14200, 73712, 365596};
  if (n < 15) {
    printf("\n========================================================\n");
    printf("Expected solutions: %d\n", expected[n]);
    printf("Result: %s\n",
           (solutions == expected[n]) ? "✓ VERIFIED" : "✗ ERROR");
  }

  return 0;
}