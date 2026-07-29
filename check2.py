import yaml

with open("data.yaml", "r") as f:
    data = yaml.safe_load(f)

print(data["names"])
