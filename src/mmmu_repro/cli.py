import argparse
from dataclasses import fields
import json
from pathlib import Path

from .common import RunLock, append_jsonl, save_json
from .config import Config, PROFILES
from .judge import JudgeConfig, JudgeClient, checked_predictions, evaluate, extraction_input, plan


def parser():
    root = argparse.ArgumentParser(description="MMMU-val Qwen free generation + Codyssey API answer extraction")
    sub = root.add_subparsers(dest="command", required=True)
    for name in ("preflight", "infer"):
        p = sub.add_parser(name)
        p.add_argument("--data-root", default="data", help="Directory containing MMMU_DEV_VAL.tsv, or its file path")
        p.add_argument("--output-dir", default="results/qwen_free3407/inference")
        p.add_argument("--limit", type=int, default=0, help="Diagnostic first-N only; 0 = complete 900. Use a different output-dir.")
        defaults = Config()
        for field in fields(Config):
            value = getattr(defaults, field.name)
            kwargs = dict(default=value, type=type(value))
            if field.name == "profile":
                kwargs["choices"] = list(PROFILES)
            p.add_argument("--"+field.name.replace("_", "-"), **kwargs)
    for name in ("judge-plan", "judge", "api-check"):
        p = sub.add_parser(name)
        if name != "api-check":
            p.add_argument("--inference-dir", default="results/qwen_free3407/inference")
            p.add_argument("--output-dir", default="results/qwen_free3407/judge")
        if name != "judge-plan":
            p.add_argument("--api-base", default=JudgeConfig.api_base)
            p.add_argument("--judge-model", default=JudgeConfig.model)
            p.add_argument("--api-key-env", default=JudgeConfig.key_env)
            p.add_argument("--judge-max-tokens", type=int, default=4096)
            p.add_argument("--reasoning-effort", default="none", choices=("none", "low", "medium", "high"))
            p.add_argument("--judge-temperature", type=float, default=0.0)
        if name == "judge":
            p.add_argument("--max-api-calls", type=int, default=900, help="Successful extraction calls in this invocation; pilot: 20")
    return root


def main():
    p = parser()
    args = p.parse_args()
    try:
        if args.command in ("preflight", "infer"):
            from .inference import preflight, run_inference
            if not 0 <= args.limit <= 900:
                raise ValueError("limit must be in 0..900")
            config = Config(**{f.name: getattr(args, f.name) for f in fields(Config)}).validate()
            with RunLock(args.output_dir):
                if args.command == "preflight":
                    _, _, result = preflight(config, args.data_root, args.output_dir, args.limit)
                    result = {k: v for k, v in result.items() if k != "items"}
                else:
                    result = run_inference(config, args.data_root, args.output_dir, args.limit)
        elif args.command == "judge-plan":
            rows = checked_predictions(args.inference_dir)
            result, prompts = plan(rows)
            out = Path(args.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            save_json(out / "plan.json", result)
            # Explicit local artifact only; no key or remote call in this command.
            save_json(out / "planned_requests.json", prompts)
        else:
            config = JudgeConfig(api_base=args.api_base, model=args.judge_model, key_env=args.api_key_env,
                                 max_tokens=args.judge_max_tokens, reasoning_effort=args.reasoning_effort,
                                 temperature=args.judge_temperature).validate()
            if args.command == "api-check":
                row = dict(question="What color is described?", choices={"A": "red", "B": "blue"},
                           question_type="multiple-choice", raw_response="It is red.", answer="A")
                response = JudgeClient(config)(extraction_input(row)[2])
                if response["text"].strip() != "A":
                    raise ValueError("Synthetic API extraction check did not return A")
                result = dict(state="ok", returned_model=response["model"], usage=response["usage"],
                              note="Paid synthetic request; no dataset sent; successful with the exact configured payload")
            else:
                if args.max_api_calls < 0:
                    raise ValueError("max-api-calls must be nonnegative")
                with RunLock(args.output_dir):
                    result = evaluate(args.inference_dir, args.output_dir, config, args.max_api_calls)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    except (ValueError, RuntimeError, OSError) as e:
        # Client transport errors are sanitized inside judge.py.
        p.exit(1, f"ERROR: {e}\n")


if __name__ == "__main__":
    main()
