using CP;

int rank = ...;
int n = ...;
int target_value = ...;
int graph[1..n, 1..n] = ...;
int n_edges = ...;
int edge_1[1..n_edges] = ...;
int edge_2[1..n_edges] = ...;

range r = 1..n;

dvar int structure[1..n] in 1..n;

subject to {
    structure[1] == 1;
    forall (i in 2..rank) ((structure[i - 1]) <= (structure[i]));
    allDifferent(structure);

    (sum (i in 1..n_edges) (graph[structure[edge_1[i]], structure[edge_2[i]]])) <= target_value;
}

execute {
    writeln(structure);
}
