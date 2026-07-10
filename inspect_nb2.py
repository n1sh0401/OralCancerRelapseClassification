import json
from pathlib import Path
p = Path("c:/Downloads/vscode/Codings/OralCancerRelapseClassification/DanishOralCancerRelapse_v2.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
for idx in range(45,56):
    cell = nb["cells"][idx]
    print("CELL", idx, cell["cell_type"])
    if cell["cell_type"] == "code":
        print("".join(cell["source"]))
    else:
        print(cell["source"])
    print("---")
