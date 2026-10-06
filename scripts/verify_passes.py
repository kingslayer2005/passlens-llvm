import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import find_llvm_tools, count_instructions
from scripts.phase3_labels import apply_pass
import pandas as pd

PASS_LIST = [
    "sroa", "instcombine", "simplifycfg", "early-cse",
    "gvn", "sccp", "adce", "dse", "reassociate", "jump-threading",
    "correlated-propagation", "licm", "loop-rotate", "indvars",
    "loop-deletion", "loop-idiom", "loop-unroll", "tailcallelim"
]

WRAPPERS = {
    "licm": "loop-mssa(licm)",
    "loop-rotate": "loop(loop-rotate)",
    "loop-deletion": "loop(loop-deletion)",
    "loop-idiom": "loop(loop-idiom)",
    "loop-unroll": "loop-unroll",
}

def main():
    tools = find_llvm_tools()
    df = pd.read_csv("data/function_index.csv")
    if "is_smoke" in df.columns:
        df = df[df["is_smoke"] == True]
        
    funcs = []
    for _, row in df.iterrows():
        funcs.append(Path(row["path"]).read_text(encoding="utf-8"))
        
    print(f"Loaded {len(funcs)} functions.")
    
    table = []
    
    for p in PASS_LIST:
        real_p = WRAPPERS.get(p, p)
        
        errs = 0
        diffs = 0
        dec = 0
        eq = 0
        inc = 0
        
        for ir in funcs:
            try:
                res = apply_pass(ir, real_p, tools)
                if res is None:
                    errs += 1
                else:
                    if res != ir:
                        diffs += 1
                    b = count_instructions(ir)
                    a = count_instructions(res)
                    if a < b: dec += 1
                    elif a == b: eq += 1
                    else: inc += 1
            except Exception as e:
                errs += 1
                
        # Try wrapping with loop
        if diffs == 0 and p not in WRAPPERS and 'loop' not in real_p:
            test_p = f"loop({p})"
            test_diffs = 0
            for ir in funcs:
                res = apply_pass(ir, test_p, tools)
                if res and res != ir: test_diffs += 1
            if test_diffs > 0:
                real_p = test_p
                errs = diffs = dec = eq = inc = 0
                for ir in funcs:
                    res = apply_pass(ir, real_p, tools)
                    if res is None: errs += 1
                    else:
                        if res != ir: diffs += 1
                        b = count_instructions(ir)
                        a = count_instructions(res)
                        if a < b: dec += 1
                        elif a == b: eq += 1
                        else: inc += 1
        
        table.append({
            "Pass": p,
            "Real Pass": real_p,
            "Errors": errs,
            "Diffs": diffs,
            "Dec": dec,
            "Eq": eq,
            "Inc": inc
        })
        
    out = pd.DataFrame(table)
    print(out.to_markdown(index=False))

if __name__ == "__main__":
    main()
