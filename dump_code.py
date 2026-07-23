import os

SKIP_DIRS = {'.venv', '.git', 'chroma_db', '__pycache__', '.pytest_cache', 'data'}
KEEP_EXT = {'.py', '.md', '.txt'}

with open('classmate_all.txt', 'w') as out:
    for root, dirs, files in os.walk('.'):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in sorted(files):
            if os.path.splitext(name)[1] in KEEP_EXT:
                path = os.path.join(root, name)
                if os.path.abspath(path) == os.path.abspath('classmate_all.txt'):
                    continue
                out.write(f"===== {path} =====\n")
                with open(path, encoding='utf-8', errors='replace') as f:
                    out.write(f.read())
                out.write("\n")

print("Done. Written to classmate_all.txt")
