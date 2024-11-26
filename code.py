import os


def extract_code_to_file(source_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as output:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.py'):  # Process only Python files
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()
                    except Exception as e:
                        code = f"# Error reading file: {file_path}\n# {e}\n"
                    # Write to the output file
                    output.write(f"\n# {'-'*40}\n")
                    output.write(f"# File: {file_path}\n")
                    output.write(f"# {'-'*40}\n")
                    output.write(code)
                    output.write("\n")


if __name__ == "__main__":
    source_directory = "app"
    output_filename = "output"

    if not os.path.exists(source_directory):
        print("Source directory does not exist. Please provide a valid path.")
    else:
        extract_code_to_file(source_directory, output_filename)
        print(f"Code has been extracted to {output_filename}")
