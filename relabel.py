import os

directory = "./datasets/chess/valid/labels"  

for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)

        with open(file_path, 'r') as file:
            lines = file.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue  
            try:
                parts[0] = str(int(parts[0]) - 1)
                new_line = " ".join(parts)
                new_lines.append(new_line)
            except ValueError:
                print(f"Warning: Skipping invalid line in {filename}: {line.strip()}")

        with open(file_path, 'w') as file:
            file.write("\n".join(new_lines) + "\n")

print("Finished adjusting class IDs.")
