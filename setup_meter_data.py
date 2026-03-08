import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


RANDOM_SEED = 42
TRAIN_RATIO = 0.9
DATASET_NAME = "meter_dataset"
SUPPORTED_AUDIO_SUFFIXES = {".mp3", ".wav"}


def annotation_candidates(annotation_dir: Path, stem: str) -> tuple[Path, ...]:
    return (
        annotation_dir / f"{stem}.beats",
        annotation_dir / f"{stem}.beats.json",
        annotation_dir / f"{stem}.beat.beats.json",
    )


def resolve_annotation_path(annotation_dir: Path, stem: str) -> Path | None:
    return next(
        (path for path in annotation_candidates(annotation_dir, stem) if path.exists()),
        None,
    )


def has_supported_annotation(annotation_dir: Path, stem: str) -> bool:
    return resolve_annotation_path(annotation_dir, stem) is not None


def dominant_meter_label(annotation_path: Path) -> str:
    if annotation_path.suffix != ".json":
        return "unknown"

    with open(annotation_path, "r", encoding="utf-8") as handle:
        annotation = json.load(handle)

    meter_counts = Counter()
    for measure in annotation.get("measures", []):
        if "time_sig_num" not in measure or "time_sig_den" not in measure:
            continue
        meter_counts[f"{int(measure['time_sig_num'])}/{int(measure['time_sig_den'])}"] += 1

    if not meter_counts:
        return "unknown"
    return meter_counts.most_common(1)[0][0]


def val_count_for_group(group_size: int) -> int:
    if group_size <= 1:
        return 0
    raw_val = int(round(group_size * (1 - TRAIN_RATIO)))
    return min(group_size - 1, max(1, raw_val))


def stratified_split(
    audio_files: list[Path], annotation_dir: Path
) -> tuple[list[Path], list[Path], dict[str, list[Path]]]:
    if not audio_files:
        raise FileNotFoundError("No annotated audio files found for meter_dataset")

    grouped_files: dict[str, list[Path]] = defaultdict(list)
    for file_path in sorted(audio_files, key=lambda path: path.stem):
        annotation_path = resolve_annotation_path(annotation_dir, file_path.stem)
        if annotation_path is None:
            continue
        grouped_files[dominant_meter_label(annotation_path)].append(file_path)

    rng = random.Random(RANDOM_SEED)
    train_files = []
    val_files = []
    for meter_label in sorted(grouped_files):
        files = grouped_files[meter_label][:]
        rng.shuffle(files)
        val_count = val_count_for_group(len(files))
        val_files.extend(files[:val_count])
        train_files.extend(files[val_count:])

    train_files.sort(key=lambda path: path.stem)
    val_files.sort(key=lambda path: path.stem)
    return train_files, val_files, grouped_files


def write_split_file(audio_files: list[Path], annotation_dir: Path, split_file_path: Path) -> None:
    train_files, val_files, grouped_files = stratified_split(audio_files, annotation_dir)

    with open(split_file_path, "w", encoding="utf-8") as handle:
        for file_path in train_files:
            handle.write(f"{file_path.stem}\ttrain\n")
        for file_path in val_files:
            handle.write(f"{file_path.stem}\tval\n")

    print(
        f"Created {split_file_path} with {len(train_files)} train and {len(val_files)} val items."
    )
    print("Stratified by dominant meter:")
    for meter_label in sorted(grouped_files):
        total = len(grouped_files[meter_label])
        val_count = val_count_for_group(total)
        train_count = total - val_count
        note = " (train only)" if total == 1 else ""
        print(f"  {meter_label}: {train_count} train / {val_count} val{note}")


def write_info_file(info_path: Path) -> None:
    info = {"has_downbeats": True}
    with open(info_path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, ensure_ascii=True, indent=2)
        handle.write("\n")
    print(f"Created {info_path}")


def update_audio_paths_csv(audio_paths_csv: Path, audio_dir: Path) -> None:
    rows = []
    if audio_paths_csv.exists():
        with open(audio_paths_csv, "r", encoding="utf-8", newline="") as handle:
            rows = [row for row in csv.reader(handle) if row]

    dataset_path = str(audio_dir.resolve().as_posix())
    updated = False
    filtered_rows = []
    for row in rows:
        if row[0] == DATASET_NAME:
            filtered_rows.append([DATASET_NAME, dataset_path])
            updated = True
        else:
            filtered_rows.append(row)

    if not updated:
        filtered_rows.append([DATASET_NAME, dataset_path])

    with open(audio_paths_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(filtered_rows)

    action = "Updated" if updated else "Appended"
    print(f"{action} {DATASET_NAME} path in {audio_paths_csv}")


def main():
    repo_root = Path(__file__).resolve().parent
    data_dir = repo_root / "data"
    dataset_dir = data_dir / "annotations" / DATASET_NAME
    audio_dir = dataset_dir / "audios"
    annotation_dir = dataset_dir / "annotations" / "beats"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"Audio directory not found: {audio_dir}")
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    audio_files = [
        file_path
        for file_path in sorted(audio_dir.glob("*"))
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_AUDIO_SUFFIXES
    ]
    missing_annotations = [
        file_path.name
        for file_path in audio_files
        if not has_supported_annotation(annotation_dir, file_path.stem)
    ]
    if missing_annotations:
        print(
            "Skipping audio files without annotations:",
            ", ".join(missing_annotations),
        )

    annotated_audio_files = [
        file_path
        for file_path in audio_files
        if has_supported_annotation(annotation_dir, file_path.stem)
    ]

    write_split_file(annotated_audio_files, annotation_dir, dataset_dir / "single.split")
    write_info_file(dataset_dir / "info.json")
    update_audio_paths_csv(data_dir / "audio_paths.csv", audio_dir)


if __name__ == "__main__":
    main()
