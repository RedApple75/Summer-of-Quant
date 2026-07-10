import json

with open("f:/Projects/SOQ Advanced Project/Project Guide Final.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("f:/Projects/SOQ Advanced Project/scratch_notebook_outline.txt", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb.get("cells", [])):
        cell_type = cell.get("cell_type", "")
        source = cell.get("source", [])
        source_str = "".join(source)
        
        if cell_type == "markdown":
            out.write(f"\n--- Cell {i} (Markdown) ---\n")
            out.write(source_str)
            out.write("\n")
        elif cell_type == "code":
            out.write(f"\n--- Cell {i} (Code Block Summary) ---\n")
            lines = source_str.split("\n")
            first_few_lines = "\n".join(lines[:5])
            out.write(first_few_lines)
            if len(lines) > 5:
                out.write("\n... [truncated] ...")
            out.write("\n")

print("Notebook parsed and outline saved to scratch_notebook_outline.txt")
