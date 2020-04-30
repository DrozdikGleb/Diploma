.PHONY: Hypercube.jar

out_dir = temp

Hypercube.jar:
	mkdir -p $(out_dir)
	javac -d $(out_dir) -encoding UTF-8 src/hypercube/*.java
	jar cfm Hypercube.jar MANIFEST.MF -C $(out_dir) hypercube -C src constraints
	rm -r -f $(out_dir)
