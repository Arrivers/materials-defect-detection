project_name = "Materials Surface Defect Detection"

defect_types = [
    "crazing",
    "inclusion",
    "patches",
    "pitted surface",
    "rolled-in scale",
    "scratches",
]

print("=" * 50)
print(project_name)
print("=" * 50)

print(f"This project contains {len(defect_types)} defect types:")

for number, defect in enumerate(defect_types, start=1):
    print(f"{number}. {defect}")

print("Project initialization completed.")