/* 
A demo of how to use MPI to implement task bag parallel pattern.

The stack implementation is ignored.

Note:

Coding environment:

    OS: Microsoft Windows 7 (64bit)
    IDE: Code::Blocks 13.12 (32bit)
    Compiler: GNU GCC Complier
    MPI: MPICH2 for Windows (32bit)
    CPU: Intel Core i7-3630QM @ 2.40GHz


Result on the coding environment (tasks per process changes each time you run this code):

Area=7583461.801487

Tasks Per Process
0	1	2	3	4
0	1649	1646	1628	1644

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "stack.h"

#define EPSILON 1e-3
#define F(arg)  cosh(arg)*cosh(arg)*cosh(arg)*cosh(arg)
#define A 0.0
#define B 5.0

#define SLEEPTIME 1

// message tag definition
#define NEW_TASK 1
#define RESULT 2
#define REQUEST_TASK 3
#define NO_TASK_AVAILABLE 4
#define TASK_TO_FINISH 5
#define JOB_FINISH 6

int *tasks_per_process;

double farmer(int);

void worker(int);

int main(int argc, char **argv ) {
  int i, myid, numprocs;
  double area, a, b;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  if(numprocs < 2) {
    fprintf(stderr, "ERROR: Must have at least 2 processes to run\n");
    MPI_Finalize();
    exit(1);
  }

  if (myid == 0) { // Farmer
    // init counters
    tasks_per_process = (int *) malloc(sizeof(int)*(numprocs));
    for (i=0; i<numprocs; i++) {
      tasks_per_process[i]=0;
    }
  }

  if (myid == 0) { // Farmer
    area = farmer(numprocs);
  } else { //Workers
    worker(myid);
  }

  if(myid == 0) {
    fprintf(stdout, "Area=%lf\n", area);
    fprintf(stdout, "\nTasks Per Process\n");
    for (i=0; i<numprocs; i++) {
      fprintf(stdout, "%d\t", i);
    }
    fprintf(stdout, "\n");
    for (i=0; i<numprocs; i++) {
      fprintf(stdout, "%d\t", tasks_per_process[i]);
    }
    fprintf(stdout, "\n");
    free(tasks_per_process);
  }
  MPI_Finalize();
  return 0;
}

double farmer(int numprocs)
{
    int i;
    double ans = 0;
    int flag;
    MPI_Status status;
    double temp[2] = {A,B}; // a temporary buffer with 2 elements
    double temp3[3] = {0, 0, 0}; // a temporary buffer with 3 elements
    int idle_workers = numprocs-1; // number of idle workers
    stack *task_bag = new_stack(); // task bag
    push(temp, task_bag); // generate first task
    while (1) {
        // see if there is a coming message
        MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
        // if a message is received
        if (flag) {
            // "new task" message
            if (status.MPI_TAG == NEW_TASK) {
                // we have two new tasks in one message here, push both tasks into bag
                MPI_Recv(temp3, 3, MPI_DOUBLE, MPI_ANY_SOURCE, NEW_TASK, MPI_COMM_WORLD, &status);
                temp[0] = temp3[0];
                temp[1] = temp3[1];
                push(temp, task_bag);
                temp[0] = temp3[1];
                temp[1] = temp3[2];
                push(temp, task_bag);
                idle_workers++;
            }
            // "result" message
            else if (status.MPI_TAG == RESULT) {
                // receive the result and add it to the sum
                MPI_Recv(temp, 2, MPI_DOUBLE, status.MPI_SOURCE, RESULT, MPI_COMM_WORLD, &status);
                ans = ans + temp[0] + temp[1];
                idle_workers++;
            }
            // "request task" message
            else if (status.MPI_TAG == REQUEST_TASK) {
                MPI_Recv(temp, 2, MPI_DOUBLE, MPI_ANY_SOURCE, REQUEST_TASK, MPI_COMM_WORLD, &status);
                // see whether the task bag is empty
                if (is_empty(task_bag)) {
                    // if task bag is empty, send "no task available" message
                    MPI_Send(temp, 2, MPI_DOUBLE, status.MPI_SOURCE, NO_TASK_AVAILABLE, MPI_COMM_WORLD);
                }
                else {
                    // task bag is not empty, send a new task to the worker
                    MPI_Send(pop(task_bag), 2, MPI_DOUBLE, status.MPI_SOURCE, TASK_TO_FINISH, MPI_COMM_WORLD);
                    // this worker get a new task to do, increase the corresponding counter
                    tasks_per_process[status.MPI_SOURCE]++;
                    idle_workers--;
                }
            }
        }
        // if no message received, check whether the job is finished
        // only when task bag is empty and all workers are idle, the job is finished
        else if ((idle_workers == numprocs-1) && (is_empty(task_bag))) {
            for (i=1;i<numprocs;i++) {
                MPI_Send(temp, 2, MPI_DOUBLE, i, JOB_FINISH, MPI_COMM_WORLD);
            }
            free_stack(task_bag);
            return ans;
        }
    }
}

void worker(int mypid)
{
    int i;
    double temp[2] = {0,0}; // a temporary buffer with 2 elements
    double temp3[3] = {0, 0, 0}; // a temporary buffer with 3 elements
    double left, mid, right, fleft, fmid, fright, larea, rarea, lrarea;
    MPI_Status status;
    while (1) {
        // send "request message" to the farmer
        MPI_Send(temp, 2, MPI_DOUBLE, 0, REQUEST_TASK, MPI_COMM_WORLD);
        // receive message from the farmer
        MPI_Recv(temp, 2, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        // wait
        usleep(SLEEPTIME);
        // receive a "job finish" message
        if (status.MPI_TAG == JOB_FINISH) {
            // worker finishes work
            break;
        }
        // receive a "no task available" message
        if (status.MPI_TAG == NO_TASK_AVAILABLE) {
            // send request again
            continue;
        }
        // receive a new task to finish
        if (status.MPI_TAG == TASK_TO_FINISH) {
            // find the area
            left = temp[0];
            right = temp[1];
            mid = (left + right) / 2;
            fleft = F(left);
            fright = F(right);
            fmid = F(mid);
            larea = (fleft + fmid) * (mid - left) / 2;
            rarea = (fmid + fright) * (right - mid) / 2;
            lrarea = (fleft + fright) * (right - left) / 2;
            // if error is not small enough
            if( fabs((larea + rarea) - lrarea) > EPSILON ) {
                // send new tasks to the farmer
                temp3[0] = left;
                temp3[1] = mid;
                temp3[2] = right;
                MPI_Send(temp3, 3, MPI_DOUBLE, 0, NEW_TASK, MPI_COMM_WORLD);
            }
            else {
                // error is small, send result to the farmer
                temp[0] = larea;
                temp[1] = rarea;
                MPI_Send(temp ,2, MPI_DOUBLE, 0, RESULT, MPI_COMM_WORLD);
            }
        }
    }
}
