package hypercube;

import java.io.*;
import java.util.ArrayList;
import java.util.Scanner;
import java.util.concurrent.ThreadLocalRandom;

public class Main {
    private static void generate(int rank, int minValue, int maxValue, String filepath) throws IOException {
        new File(filepath.substring(0, filepath.indexOf('/'))).mkdirs();

        System.out.println(rank + " " + minValue + " " + maxValue + "\n");

        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();
        int n = 1;
        for (int i = 0; i < rank; i++) {
            n *= 2;
        }
        for (int i = 0; i < n; i++) {
            graph.add(new ArrayList<>());
            for (int j = 0; j < n; j++) {
                graph.get(i).add(0);
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int val = ThreadLocalRandom.current().nextInt(minValue, Math.min(maxValue + 1, Integer.MAX_VALUE));
                graph.get(i).set(j, val);
                graph.get(j).set(i, val);
            }
        }

        try (FileWriter res = new FileWriter(filepath)) {
            res.write(rank + " " + n + "\n");
            for (ArrayList<Integer> a : graph) {
                for (Integer val : a) {
                    res.write(val + " ");
                }
                res.write("\n");
            }
        }
    }

    private static void makeDznFile(long targetValue, String input, String output) throws IOException {
        makeDznFile(targetValue, input, output, true);
    }

    private static void makeDznFile(long targetValue, String input, String output, boolean isForMinizinc) throws IOException {
        int rank, n;
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();

        try (FileReader reader = new FileReader(input)) {
            Scanner scanner = new Scanner(reader);
            rank = scanner.nextInt();
            n = scanner.nextInt();

            for (int i = 0; i < n; i++) {
                graph.add(new ArrayList<>());
                for (int j = 0; j < n; j++) {
                    graph.get(i).add(j, scanner.nextInt());
                }
            }
        }
        try (FileWriter writer = new FileWriter(output)) {
            writer.write("rank = " + rank + ";\n");
            writer.write("n = " + n + ";\n");
            writer.write("target_value = " + targetValue + ";\n");

            if (isForMinizinc) {
                writer.write("graph = array2d(1.." + n + ", 1.." + n + ", [");
                boolean isFirst = true;
                for (ArrayList<Integer> a : graph) {
                    for (Integer val : a) {
                        if (! isFirst) {
                            writer.write(", ");
                        } else {
                            isFirst = false;
                        }
                        writer.write(val + "");
                    }
                }
                writer.write("]);\n");
            } else {
                writer.write("graph = [");

                for (int i = 0; i < graph.size(); i++) {
                    if (i > 0) {
                        writer.write(", ");
                    }
                    writer.write("[");
                    boolean isFirst = true;
                    for (Integer val : graph.get(i)) {
                        if (! isFirst) {
                            writer.write(", ");
                        } else {
                            isFirst = false;
                        }
                        writer.write(val + "");
                    }
                    writer.write("]");
                }
                writer.write("];\n");
            }

            ArrayList<String> edge1 = new ArrayList<>(), edge2 = new ArrayList<>();

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < rank; j++) {
                    if ((i ^ (1 << j)) > i) {
                        edge1.add(i              + 1 + "");
                        edge2.add((i ^ (1 << j)) + 1 + "");
                    }
                }
            }
            writer.write("n_edges = " + edge1.size() + ";\n");
            writer.write("edge_1 = [" + String.join(", ", edge1) + "];\n");
            writer.write("edge_2 = [" + String.join(", ", edge2) + "];\n");
        }
    }

    private static String resolveResult(String output, int rank, ArrayList<ArrayList<Integer>> graph) {
        if (output.contains("no solution")) {
            return "UNSATISFIABLE";
        }
        if (output.contains("UNSATISFIABLE") || output.equals("TL")) {
            return output;
        }

        int pos1 = output.indexOf("[");
        int pos2 = output.indexOf("]");
        ArrayList<Integer> structure = new ArrayList<>();
        System.out.println(output);
        for (String val : output.substring(pos1 + 1, pos2).trim().split(" ")) {
            if (!val.trim().isEmpty()) {
                structure.add(Integer.valueOf(val.trim()));
            }
        }
        long res = 0;

        for (int i = 0; i < structure.size(); i++) {
            for (int j = 0; j < rank; j++) {
                if ((i ^ (1 << j)) > i) {
                    res += graph.get(structure.get(i) - 1).get(structure.get(i ^ (1 << j)) - 1);
                }
            }
        }
        StringBuilder sb = new StringBuilder(res + " [");
        for (int i = 0; i < structure.size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(structure.get(i));
        }
        sb.append("]");
        return sb.toString();
    }

    private static String searchStep(long maxRes, long minRes, long mid, String input, String tempDir, String oplSolver
                                   , int timeLimitS, int rank, ArrayList<ArrayList<Integer>> graph) throws IOException {
        long currentTimeMillis = System.currentTimeMillis();

        System.out.println("Possible values in [" + maxRes + ", " + minRes + "], testing " + mid + " :");
        makeDznFile(mid, input, tempDir + "/data.dzn", oplSolver.isEmpty());
        Solver solver = new Solver();
        String res = solver.solve(timeLimitS, tempDir, tempDir + "/data.dzn", oplSolver);
        if (!oplSolver.isEmpty()) {
            res = resolveResult(res, rank, graph);
        }
        long currentTimeMillis2 = System.currentTimeMillis();

        System.out.print("  " + (currentTimeMillis2 - currentTimeMillis) + " ms : ");
        return res;
    }

    private static void search(String input, String tempDir, int timeLimitS, String oplSolver) throws IOException {
        int rank, n;
        ArrayList<ArrayList<Integer>> graph = new ArrayList<>();

        long maxRes = 0, minRes = 0;

        try (FileReader reader = new FileReader(input)) {
            Scanner scanner = new Scanner(reader);
            rank = scanner.nextInt();
            n = scanner.nextInt();

            for (int i = 0; i < n; i++) {
                graph.add(new ArrayList<>());
                ArrayList<Integer> a = new ArrayList<>();

                for (int j = 0; j < n; j++) {
                    int val = scanner.nextInt();
                    graph.get(i).add(val);
                    a.add(val);
                }
                a.sort(Integer::compareTo);

                for (int j = 0; j < rank; j++) {
                    minRes += a.get(j);
                    maxRes += a.get(a.size() - j - 1);
                }
            }
        }
        minRes /= 2;
        maxRes /= 2;
        String bestResult;
        {
            StringBuilder sb = new StringBuilder(maxRes + " [");
            for (int i = 1; i <= n; i++) {
                if (i > 1) {
                    sb.append(", ");
                }
                sb.append(i);
            }
            bestResult = sb.toString();
        }

        long leftLim = maxRes, rightLim = minRes;

        while (leftLim > rightLim) {
            long mid = (leftLim + rightLim) / 2;

            String res = searchStep(maxRes, minRes, mid, input, tempDir, oplSolver, timeLimitS, rank, graph);

            if (res.contains("UNSATISFIABLE") || res.equals("TL")) {
                rightLim = mid + 1;
            } else {
                maxRes = leftLim = Integer.valueOf(res.split(" ")[0]);
                bestResult = res;
                try (FileWriter writer = new FileWriter(tempDir + "/result.txt")) {
                    writer.write(bestResult);
                }
            }
            System.out.println(res);
        }

        leftLim++;
        rightLim = minRes;

        while (leftLim > rightLim) {
            long mid = (leftLim + rightLim) / 2;

            String res = searchStep(maxRes, minRes, mid, input, tempDir, oplSolver, timeLimitS, rank, graph);

            if (res.contains("UNSATISFIABLE")) {
                minRes = rightLim = mid + 1;
            } else {
                leftLim = mid;
            }
            System.out.println(res);
        }
        System.out.println("Possible values in [" + maxRes + ", " + minRes + "], best result : " + bestResult);
    }

    public static void main(String[] args) {
        try {
            if (args.length == 0) {
                System.out.println("Недостаточно аргументов.");
                return;
            }
            if (args[0].equals("generate")) {
                if (args.length < 5) {
                    System.out.println("Недостаточно аргументов.");
                    return;
                }
                generate(Integer.valueOf(args[1]), Integer.valueOf(args[2]), Integer.valueOf(args[3]), args[4]);
                return;
            }
            if (args[0].equals("make_dzn")) {
                if (args.length < 4) {
                    System.out.println("Недостаточно аргументов.");
                    return;
                }
                makeDznFile(Integer.valueOf(args[1]), args[2], args[3]);
                return;
            }
            if (args[0].equals("solve")) {
                if (args.length < 4) {
                    System.out.println("Недостаточно аргументов.");
                    return;
                }
                Solver solver = new Solver();
                String output = solver.solve(Integer.valueOf(args[1]), args[2], args[3], "");
                System.out.println(output);
                return;
            }
            if (args[0].equals("search")) {
                if (args.length < 4) {
                    System.out.println("Недостаточно аргументов.");
                    return;
                }
                String oplSolver = "";
                if (args.length == 5) {
                    oplSolver = args[4];
                }
                search(args[1], args[2], Integer.valueOf(args[3]), oplSolver);
                return;
            }
            System.out.println("Неизвестный аргумент \"" + args[0] + "\"");
        } catch (IOException ex) {
            System.out.println(ex.toString());
        }
    }
}
