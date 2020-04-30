include "globals.mzn";

int : rank;
int : n;
int : target_value;
array [1..n, 1..n] of int : graph;
int : n_edges;
array [1..n_edges] of 1..n : edge_1;
array [1..n_edges] of 1..n : edge_2;

array [1..n] of var 1..n : structure;

constraint structure[1] == 1;
constraint forall (i in 2..rank) (structure[i - 1] < structure[i]);

constraint alldifferent(structure);

constraint (sum (i in 1..n_edges) (graph[structure[edge_1[i]], structure[edge_2[i]]])) <= target_value;
solve satisfy;

output [show(sum (i in 1..n_edges) (graph[structure[edge_1[i]], structure[edge_2[i]]])), " ", show(structure), "\n"];
