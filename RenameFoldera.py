import os

#os.chdir("C/Users/ASUS/PycharmProjects/pythonProject25")
# Set the path to the folder containing the subfolders
path = "Dataset100/train"

# Loop through the folders and rename them
for i in range(1, 56):
    old_name = os.path.join(path, str(i))
    new_name = os.path.join(path, str(i - 1))
    os.rename(old_name, new_name)

print("Folders renamed successfully.")