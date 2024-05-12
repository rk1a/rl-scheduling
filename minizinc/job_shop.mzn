int: num_jobs;
int: num_tasks;
int: num_machines;
array [JOB,TASK] of int: durations;
array [JOB,TASK] of int: machine_ids;

set of int: JOB = 1..num_jobs;
set of int: TASK = 1..num_tasks;
set of int: MACHINE = 1..num_machines;
int: last = num_tasks; %% index of last task
int: max_time = sum(i in JOB, j in TASK)(durations[i,j]); %% worst case makespan

%% decision variables
array [JOB,TASK] of var 0..max_time: start_times;
var 0..max_time: end;

%% ensure the tasks of a job occur in sequence
constraint
    forall(i in JOB) (
        forall(j in TASK where j < last) 
            (start_times[i,j] + durations[i,j] <= start_times[i,j+1]) /\
        start_times[i,last] + durations[i,last] <= end
    );

 %% ensure no overlap of tasks that run on the same machine
constraint
    forall(i,k in JOB where i < k) (
        forall(j, l in TASK where machine_ids[i, j] == machine_ids[k, l]) (
            start_times[i,j] + durations[i,j] <= start_times[k,l] \/
            start_times[k,l] + durations[k,l] <= start_times[i,j]
        )
    );

solve minimize end;

int: digs = ceil(log(10.0,int2float(max_time))); % digits for output
output ["end = \(end)\n"] ++
       [ show_int(digs,start_times[i,j]) ++ " " ++ 
         if j == last then "\n" else "" endif |
         i in JOB, j in TASK ];

output [show_gantt([start_times[i, j] | i in JOB, j in TASK], [durations[i, j] | i in JOB, j in TASK]), "\n"];