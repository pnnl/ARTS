from pathlib import Path

log_files = list(Path('build/examples/cpu/').glob('*.log'))

counter = 0
for f in log_files:
    contents = open(f, "r").read().split("\n")
    for c in contents:
        if ("Fib 25: 75025" in c):
            counter += 1

print(f"Total runs: {len(log_files)}")
print(f"Successful runs: {counter}")

if (len(log_files) == counter):
    print("Runs successful!")
else:
    print("Runs failed!")