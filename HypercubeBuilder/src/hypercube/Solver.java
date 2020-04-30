package hypercube;

import java.io.*;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import java.util.stream.Collectors;

public class Solver {
    private final Lock lock = new ReentrantLock();
    private final Condition condition = lock.newCondition();
    private Boolean isDestroyed;

    public String solve(int timeLimitS, String tempDir, String minizincData, String oplSolver) throws IOException {
        new File(tempDir).mkdirs();

        String output = "";
        String extension = (oplSolver.isEmpty()) ? ".mzn" : ".mod";
        String constraints = tempDir + "/constraints" + extension;
        try {
            try (FileWriter res = new FileWriter(constraints)) {
                BufferedReader reader = new BufferedReader(new InputStreamReader(Main.class.getResourceAsStream("/constraints/constraints" + extension)));
                String line;
                while ((line = reader.readLine()) != null) {
                    res.write(line + "\n");
                }
            }

            lock.lock();
            isDestroyed = false;
            lock.unlock();

            ProcessBuilder pb;

            if (oplSolver.isEmpty()) {

                pb = new ProcessBuilder("minizinc",
                        "--solver", "Chuffed",
                        constraints, minizincData);
            } else {
                System.out.println(oplSolver + " " + constraints + " " + minizincData);
                pb = new ProcessBuilder(oplSolver, constraints, minizincData);
            }
            final Process process = pb.start();
            
            {
                ProcessBuilder lPB = new ProcessBuilder("pgrep", "fzn-chuffed");
                Process p = lPB.start();
                BufferedReader br = new BufferedReader(new InputStreamReader(p.getInputStream()));
                p.waitFor();
            }

            final BufferedReader br = new BufferedReader(new InputStreamReader(process.getInputStream()));

            Thread killer = new Thread(() -> {
                lock.lock();
                long start = System.currentTimeMillis();
                try {
                    while (process.isAlive()) {
                        long currentTimeMillis = System.currentTimeMillis();

                        if (currentTimeMillis - start > timeLimitS * 1000) {
                            process.destroyForcibly();
                            isDestroyed = true;
                            try {
                                BufferedReader bufferedReader;
                                {
                                    ProcessBuilder lPB = new ProcessBuilder("pgrep", "fzn-chuffed");
                                    Process p = lPB.start();
                                    bufferedReader = new BufferedReader(new InputStreamReader(p.getInputStream()));
                                    p.waitFor();
                                }

                                for (String val : bufferedReader.lines().collect(Collectors.joining(" ")).trim().split(" ")) {
                                    if (! val.isEmpty()) {
//                                        System.out.println("\"" + val + "\"");

                                        ProcessBuilder lPB = new ProcessBuilder("kill", val);
                                        Process p = lPB.start();
                                        p.waitFor();

                                    }
                                }
                            } catch (IOException ex) {
                                System.out.println("\"" + ex + "\" in killer");
                            }
                            break;
                        }
                        condition.await(timeLimitS * 1000 - (currentTimeMillis - start), TimeUnit.MILLISECONDS);
                    }
                } catch (InterruptedException ex) {
                    isDestroyed = true;
                    process.destroyForcibly();
                } finally {
                    lock.unlock();
                }
            });

            killer.start();
            output = br.lines().collect(Collectors.joining("\n"));
            int exitCode = process.waitFor();

            lock.lock();
            try {
                condition.signal();
            } finally {
                lock.unlock();
            }
            killer.join();

            lock.lock();
            if (isDestroyed) {
                output = "TL";
            }
            lock.unlock();
        } catch (InterruptedException ex) {
            System.out.println("Fatal error : " + ex);
        }
        return output;
    }
}
