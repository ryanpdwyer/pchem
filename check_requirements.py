
import subprocess

lines = open("requirements.txt", "r").readlines()

packages = [line.split("==")[0] for line in lines]

def in_packages(x):
    return any(package in x for package in packages)

result = subprocess.run(["pip", "list"], capture_output=True, encoding='utf-8')

lines = [x for x in result.stdout.split("\n") if in_packages(x)]




lines_dict = {}
for l in lines:
    key, val = l.split()
    lines_dict[key] = val


lines_together = [f"{package}=={lines_dict[package]}" for package in packages]

print("\n".join(lines_together))