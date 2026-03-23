#!/usr/bin/env python3
"""
MixSense CLI — run the chemistry pipeline without the Gradio UI.

Modes:
  chat      Interactive chat with the chemistry agent (requires HF_TOKEN)
  resolve   Resolve chemical names to SMILES
  predict   Predict reaction products from reactants + reagents
  lookup    Look up NMR reference spectra by SMILES
  pipeline  Run the full pipeline: resolve → predict → lookup (no LLM needed)

Usage:
  uv run python -m app.cli resolve "anisole" "Br2" "FeBr3"
  uv run python -m app.cli predict --reactants "COc1ccccc1 . BrBr" --reagents "Br[Fe](Br)Br"
  uv run python -m app.cli lookup "COc1ccccc1" "c1ccccc1"
  uv run python -m app.cli pipeline --reactants "anisole" "Br2" --reagents "FeBr3"
  uv run python -m app.cli chat "What products do I get from brominating anisole?"
"""

import argparse
import json
import sys


def cmd_resolve(args):
    from app.tools_reactiont5 import resolve_names_to_smiles

    for name in args.names:
        result = resolve_names_to_smiles(name)
        if result:
            print(f"  {name:25} → {result[0]}")
        else:
            print(f"  {name:25} → ERROR: could not resolve")


def cmd_predict(args):
    from app.tools_reactiont5 import propose_products, get_unique_components

    print(f"Reactants: {args.reactants}")
    print(f"Reagents:  {args.reagents or '(none)'}")
    print()

    predictions = propose_products(args.reactants, reagents=args.reagents or "", n_best=args.n_best)

    if not predictions:
        print("No predictions returned.")
        return

    print(f"Top-{len(predictions)} predictions:")
    for i, (smiles, score) in enumerate(predictions, 1):
        print(f"  {i}. {smiles}  (score: {score:.3f})")

    unique = get_unique_components([s for s, _ in predictions])
    print(f"\nUnique products: {unique}")

    if args.json:
        print(json.dumps({"predictions": [{"smiles": s, "score": sc} for s, sc in predictions], "unique": unique}, indent=2))


def cmd_lookup(args):
    from app.tools_nmrbank import get_reference_by_smiles

    for smiles in args.smiles:
        ref = get_reference_by_smiles(smiles)
        if ref:
            n_peaks = len(ref.get("ppm", []))
            ppm = ref.get("ppm", [])[:5]
            print(f"  {smiles:25} → {ref.get('name', '?'):20} {n_peaks} peaks  ppm={ppm}{'...' if n_peaks > 5 else ''}")
        else:
            print(f"  {smiles:25} → NOT FOUND in NMRBank")


def cmd_pipeline(args):
    from app.tools_reactiont5 import resolve_names_to_smiles, propose_products, get_unique_components
    from app.tools_nmrbank import get_reference_by_smiles

    # Step 1: resolve names
    print("Step 1: Name Resolution")
    print("-" * 40)
    reactant_smiles = []
    for name in args.reactants:
        result = resolve_names_to_smiles(name)
        if result:
            print(f"  {name:20} → {result[0]}")
            reactant_smiles.append(result[0])
        else:
            print(f"  {name:20} → ERROR: could not resolve (skipping)")

    reagent_smiles = []
    for name in (args.reagents or []):
        result = resolve_names_to_smiles(name)
        if result:
            print(f"  {name:20} → {result[0]} (reagent)")
            reagent_smiles.append(result[0])
        else:
            print(f"  {name:20} → ERROR: could not resolve reagent (skipping)")

    if not reactant_smiles:
        print("No reactants could be resolved. Aborting.")
        sys.exit(1)

    # Step 2: predict products
    print("\nStep 2: Reaction Prediction")
    print("-" * 40)
    reactants_str = " . ".join(reactant_smiles)
    reagents_str = " . ".join(reagent_smiles)
    print(f"  Reactants: {reactants_str}")
    print(f"  Reagents:  {reagents_str or '(none)'}")

    predictions = propose_products(reactants_str, reagents=reagents_str, n_best=5)
    if predictions:
        print(f"\n  Top products:")
        for i, (smiles, score) in enumerate(predictions[:3], 1):
            print(f"    {i}. {smiles}  (score: {score:.3f})")
        unique_products = get_unique_components([s for s, _ in predictions])
    else:
        print("  No predictions returned.")
        unique_products = []

    # Step 3: reference lookup
    all_species = reactant_smiles + unique_products
    print(f"\nStep 3: NMR Reference Lookup")
    print("-" * 40)
    found_refs = []
    for smiles in all_species:
        ref = get_reference_by_smiles(smiles)
        if ref:
            n_peaks = len(ref.get("ppm", []))
            print(f"  ✓ {smiles:30} → {ref.get('name', '?'):20} ({n_peaks} peaks)")
            found_refs.append(ref)
        else:
            print(f"  ✗ {smiles:30} → not in NMRBank")

    print(f"\nSummary: {len(found_refs)}/{len(all_species)} species found in NMRBank")

    if args.json:
        output = {
            "reactants": dict(zip(args.reactants, reactant_smiles)),
            "predictions": [{"smiles": s, "score": sc} for s, sc in predictions],
            "references": [{"name": r.get("name"), "smiles": r.get("smiles"), "n_peaks": len(r.get("ppm", []))} for r in found_refs],
        }
        print("\n" + json.dumps(output, indent=2))


def cmd_chat(args):
    from app.chemistry_agent import ChemistryAgent

    agent = ChemistryAgent(model_name=args.model)

    if args.message:
        # Single-shot mode
        print(f"Query: {args.message}\n")
        response, tool_calls = agent.run_sync(args.message)
        if tool_calls:
            print("Tool calls:")
            for tc in tool_calls:
                print(f"  {tc['tool']}({json.dumps(tc['arguments'])[:80]})")
                result_str = json.dumps(tc['result'])[:120]
                print(f"    → {result_str}")
            print()
        print(f"Response:\n{response}")
    else:
        # Interactive mode
        print(f"MixSense Chat (model: {args.model}). Type 'quit' to exit.\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye.")
                break
            if user_input.lower() in ("quit", "exit", "q"):
                break
            if not user_input:
                continue

            for response, tool_calls in agent.run(user_input):
                if tool_calls:
                    for tc in tool_calls:
                        print(f"  [tool] {tc['tool']}({json.dumps(tc['arguments'])[:60]}...)")
            print(f"Agent: {response}\n")


def main():
    parser = argparse.ArgumentParser(
        prog="mixsense",
        description="MixSense CLI — NMR chemistry analysis pipeline",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    # resolve
    p_resolve = sub.add_parser("resolve", help="Resolve chemical names to SMILES")
    p_resolve.add_argument("names", nargs="+", help="Chemical names")

    # predict
    p_predict = sub.add_parser("predict", help="Predict reaction products")
    p_predict.add_argument("--reactants", required=True, help="Reactant SMILES joined by ' . '")
    p_predict.add_argument("--reagents", default="", help="Reagent SMILES")
    p_predict.add_argument("--n-best", type=int, default=5)
    p_predict.add_argument("--json", action="store_true")

    # lookup
    p_lookup = sub.add_parser("lookup", help="Look up NMR references by SMILES")
    p_lookup.add_argument("smiles", nargs="+", help="SMILES strings")

    # pipeline
    p_pipeline = sub.add_parser("pipeline", help="Run full pipeline: resolve → predict → lookup")
    p_pipeline.add_argument("--reactants", nargs="+", required=True, help="Reactant names")
    p_pipeline.add_argument("--reagents", nargs="*", help="Reagent names")
    p_pipeline.add_argument("--json", action="store_true")

    # chat
    p_chat = sub.add_parser("chat", help="Chat with the LLM agent (requires HF_TOKEN)")
    p_chat.add_argument("message", nargs="?", help="Single message (omit for interactive mode)")
    p_chat.add_argument("--model", default="DeepSeek V3", help="LLM model name")

    args = parser.parse_args()

    dispatch = {
        "resolve": cmd_resolve,
        "predict": cmd_predict,
        "lookup": cmd_lookup,
        "pipeline": cmd_pipeline,
        "chat": cmd_chat,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
