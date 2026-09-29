import ast
import base64
from collections import Counter
import csv
import hashlib
import io
from pathlib import Path
import string

from .common import file_hash, digest, save_json

DATA_URL = "https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv"
# The public host's TLS certificate was expired during packaging. The same
# credential-free HTTP artifact is accepted ONLY if its pinned SHA256 matches.
FALLBACK_URL = "http://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv"
FALLBACK_SHA256 = "cae0445a60c2a22f29cb9df43d2d97671baa8d5921df5d8667f4f3b1f643a478"
KNOWN_MD5 = {
    "521afc0f3bf341e6654327792781644d": "Qwen pinned dataset_utils.py",
    "585e8ad75e73f75dcad265dfd0417d64": "VLMEvalKit MMMUDataset dataset revision",
}


def download_data(path):
    import requests
    path = Path(path)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".download")
    print(f"Downloading public MMMU TSV to {path}", flush=True)
    for url in (DATA_URL, FALLBACK_URL):
        try:
            with requests.get(url, stream=True, timeout=(15, 180)) as response:
                response.raise_for_status()
                with temp.open("wb") as f:
                    for block in response.iter_content(1024 * 1024):
                        f.write(block)
        except requests.RequestException:
            if url == DATA_URL:
                print("HTTPS download unavailable; trying the same public artifact with a pinned SHA256 check", flush=True)
                continue
            raise RuntimeError("Dataset download unavailable. Supply a verified local MMMU_DEV_VAL.tsv using --data-root") from None
        md5 = file_hash(temp, "md5")
        sha256 = file_hash(temp)
        if md5 not in KNOWN_MD5 or (url == FALLBACK_URL and sha256 != FALLBACK_SHA256):
            raise ValueError("Downloaded TSV checksum is unknown. Inspect the upstream revision; file kept as .download")
        temp.replace(path)
        save_json(path.with_suffix(".source.json"), dict(url=url, md5=md5, sha256=sha256))
        return path


def nonempty(value):
    return value is not None and str(value).strip().lower() not in ("", "nan")


def list_value(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    if not nonempty(value):
        return []
    text = str(value)
    if text.startswith("[") and text.endswith("]"):
        parsed = ast.literal_eval(text)
        if not isinstance(parsed, list):
            raise ValueError("Expected image list")
        return [str(x) for x in parsed]
    return [text]


def choices_of(row):
    return {k: str(row[k]) for k in string.ascii_uppercase if nonempty(row.get(k))}


def validate_coverage(rows):
    ids = [r["id"] for r in rows]
    types = Counter(r["question_type"] for r in rows)
    subjects = Counter(r["subject"] for r in rows)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs")
    if len(rows) != 900 or types != {"multiple-choice": 847, "open": 53}:
        raise ValueError(f"Expected MMMU validation 900=847 MC+53 open, got {len(rows)}, {dict(types)}")
    if len(subjects) != 30 or set(subjects.values()) != {30}:
        raise ValueError(f"Expected 30 subjects x 30 questions, got {dict(subjects)}")
    return dict(n=900, question_types=dict(types), subjects=dict(subjects), ids_sha256=digest(ids))


def load_rows(path):
    path = Path(path)
    checksum = file_hash(path, "md5")
    if checksum not in KNOWN_MD5:
        raise ValueError(f"Unrecognized TSV MD5 {checksum}. Use the version linked by Qwen/VLMEvalKit; no silent data substitution.")
    # Embedded images can be much larger than csv's default field limit.
    csv.field_size_limit(256 * 1024 * 1024)
    with path.open(encoding="utf-8-sig", newline="") as f:
        source = list(csv.DictReader(f, delimiter="\t"))
    if not source or not {"index", "question", "answer", "split"}.issubset(source[0]):
        raise ValueError("Missing required MMMU_DEV_VAL columns")
    if len({str(r['index']) for r in source}) != len(source):
        raise ValueError("Duplicate TSV index")
    image_map = {str(r["index"]): r.get("image", "") for r in source}
    selected = [r for r in source if r["split"].strip().lower() in ("val", "validation")]
    subject_key = next((k for k in ("category", "l2-category", "subject")
                        if len(Counter(r.get(k, "") for r in selected)) == 30
                        and set(Counter(r.get(k, "") for r in selected).values()) == {30}), None)
    if not subject_key:
        raise ValueError("Cannot identify the 30 balanced subject categories")
    rows = []
    for row in selected:
        row = dict(row)
        image = row.get("image", "")
        seen = set()
        while nonempty(image) and len(image) <= 64:
            if image in seen or image not in image_map:
                raise ValueError("Broken/cyclic image reference")
            seen.add(image)
            image = image_map[image]
        images = list_value(image)
        if not images:
            raise ValueError(f"Missing embedded images for {row['index']}")
        choices = choices_of(row)
        kind = "multiple-choice" if choices else "open"
        answer = row["answer"]
        if not nonempty(answer) or (choices and answer not in choices):
            raise ValueError(f"Invalid reference answer for {row['index']}")
        row.update(id=str(row["index"]), subject=row[subject_key], question_type=kind,
                   choices=choices, images=images)
        rows.append(row)
    # Match Qwen's source dataset order; never choose questions based on correctness.
    coverage = validate_coverage(rows)
    return rows, dict(coverage, dataset_file_sha256=file_hash(path), dataset_md5=checksum,
                      dataset_revision_note=KNOWN_MD5[checksum], subject_column=subject_key,
                      source_rows=len(source), selected_split="validation")


def text_prompt(row):
    # Same strings and whitespace as pinned Qwen build_mmmu_prompt, no added CoT.
    text = f"Hint: {row['hint']}\n" if nonempty(row.get("hint")) else ""
    text += f"Question: {row['question']}\n"
    choices = row["choices"]
    if choices:
        text += "Options:\n" + "".join(f"{k}. {v}\n" for k, v in choices.items())
        text += "Please select the correct answer from the options above. \n"
    return text.rstrip()


def messages_for(row, directory, config):
    from PIL import Image
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len(row["images"]) > config.max_images:
        raise ValueError(f"Too many images for {row['id']}")
    paths = list_value(row.get("image_path"))
    multi = len(row["images"]) > 1
    if multi and len(paths) != len(row["images"]):
        raise ValueError("Image/image_path count mismatch")
    content, hashes = [], []
    for i, payload in enumerate(row["images"]):
        raw = base64.b64decode(payload, validate=True)
        image_hash = hashlib.sha256(raw).hexdigest()
        # Preserve Qwen image encoding: single images saved as jpg, multi retain suffix.
        suffix = Path(paths[i]).suffix.lower() if multi else ".jpg"
        if suffix not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            raise ValueError("Unsupported image suffix")
        target = directory / f"{image_hash}{suffix}"
        # Rewrite from immutable source bytes, never trust an old cached image.
        with Image.open(io.BytesIO(raw)) as im:
            im.save(target)
        content.append(dict(type="image", image=str(target.resolve()),
                            min_pixels=config.min_pixels, max_pixels=config.max_pixels))
        hashes.append(file_hash(target))
    content.append(dict(type="text", text=text_prompt(row)))
    return [dict(role="user", content=content)], hashes
