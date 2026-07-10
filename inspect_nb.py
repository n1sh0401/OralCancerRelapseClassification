import json
from pathlib import Path
p = Path("c:/Downloads/vscode/Codings/OralCancerRelapseClassification/DanishOralCancerRelapse_v2.ipynb")
nb = json.loads(p.read_text(encoding="utf-8"))
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "joblib.dump" in src:
            print(f"CELL {idx}")
            print(src)
            print("---")
