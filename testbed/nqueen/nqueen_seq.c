#include <bits/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

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

// Parallel backtracking function using OpenMP tasks
void solveParallel(int board[], int row, int n, int *local_count,
                   int *task_count) {
  if (row == n) {
    (*local_count)++;
    return;
  }

  for (int col = 0; col < n; col++) {
    if (isSafe(board, row, col)) {

      // Create a task for each valid placement
      int *new_board = (int *)malloc(n * sizeof(int));
      // Copy the current board state
      memcpy(new_board, board, row * sizeof(int));
      new_board[row] = col;

      solveParallel(new_board, row + 1, n, local_count, task_count);

      free(new_board);
    }
  }
}

// Main solving function that sets up the parallel region
int solveNqueens(int n, int *total_tasks) {
  int solution_count = 0;
  int task_count = 0;
  int *board = (int *)malloc(n * sizeof(int));

  printf("Starting N-Queens solver with 1 threads\n");

  // Start the parallel solving process
  solveParallel(board, 0, n, &solution_count, &task_count);

  free(board);
  return solution_count;
}

int main(int argc, char *argv[]) {
  int n = 8; // Default board size

  // Parse command line arguments
  if (argc > 1) {
    n = atoi(argv[1]);
    if (n < 1 || n > 20) {
      printf("Board size must be between 1 and 20\n");
      return 1;
    }
  }

  printf("Solving %d-Queens problem\n", n);
  printf("Using OpenMP 3.0 task parallelism\n");

  struct timespec start, end;
  clock_gettime(CLOCK_REALTIME, &start);
  double start_time = start.tv_sec + start.tv_nsec / 1e9;

  printf("Using task parallel approach\n");
  int total_tasks = 0;
  int solutions = solveNqueens(n, &total_tasks);

  clock_gettime(CLOCK_REALTIME, &end);
  double end_time = end.tv_sec + end.tv_nsec / 1e9;

  printf("\nResults:\n");
  printf("Number of solutions: %d\n", solutions);
  printf("Total tasks generated: %d\n", total_tasks);
  printf("Execution time: %.4f seconds\n", end_time - start_time);

  // Print expected results for verification
  int expected[] = {0,  1,   0,   0,    2,     10,    4,     40,
                    92, 352, 724, 2680, 14200, 73712, 365596};
  if (n < 15) {
    printf("Expected solutions for %d-queens: %d\n", n, expected[n]);
    if (solutions == expected[n]) {
      printf("✓ Result verified!\n");
    } else {
      printf("✗ Result incorrect!\n");
    }
  }

  return 0;
}
