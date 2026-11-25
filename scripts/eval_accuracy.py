"""Accuracy & confusion-matrix evaluation runner for VoiceSuraksha-Sync (DetectifAI Eidos + Aletheia).

Usage:
  python scripts/eval_accuracy.py --dataset-dir ./eval_wavs --speaker-id SUB-EVAL-01 --output reports/eval_accuracy.json

Expected folder structure inside --dataset-dir:
  genuine/    (genuine enrolled speaker samples; optional accent subfolders e.g. en-IN, hi-IN, kn-IN, hinglish)
  noisy/      (genuine speaker under noisy conditions; treated as expected-accept for FRR breakdown)
  cloned/     (synthetic / AI voice clone samples; expected-block)
  impostor/   (different real human speaker samples; expected-block)

Never invents metrics: writes measured FAR, FRR, confusion matrix, and per-category counts from real runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.detectif_client import DetectifClient, DetectifProviderError  # noqa: E402
from app.policy import VoiceEvidenceInput, SessionPolicyInput, evaluate_voice_authorization  # noqa: E402

POSITIVE_LABELS = {"genuine", "noisy"}
NEGATIVE_LABELS = {"cloned", "impostor"}
ALL_LABELS = POSITIVE_LABELS | NEGATIVE_LABELS


def evaluate_labeled_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute exact FAR, FRR, 2x2 confusion matrix, and category/accent breakdown from labeled results."""
    genuine_total = 0
    genuine_accepted = 0
    genuine_rejected_or_held = 0
    attack_total = 0
    attack_blocked = 0
    attack_accepted = 0
    inconclusive_count = 0

    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "accepted": 0, "blocked_or_held": 0})
    by_accent: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "accepted": 0, "blocked_or_held": 0})

    for rec in records:
        label = str(rec.get("label", "")).lower()
        accent = str(rec.get("accent") or "unspecified")
        outcome = str(rec.get("outcome", ""))
        accepted = outcome == "APPROVED"

        by_category[label]["total"] += 1
        by_accent[accent]["total"] += 1
        if accepted:
            by_category[label]["accepted"] += 1
            by_accent[accent]["accepted"] += 1
        else:
            by_category[label]["blocked_or_held"] += 1
            by_accent[accent]["blocked_or_held"] += 1

        if outcome == "REVIEW / HOLD":
            inconclusive_count += 1

        if label in POSITIVE_LABELS:
            genuine_total += 1
            if accepted:
                genuine_accepted += 1
            else:
                genuine_rejected_or_held += 1
        elif label in NEGATIVE_LABELS:
            attack_total += 1
            if accepted:
                attack_accepted += 1
            else:
                attack_blocked += 1

    far = round(attack_accepted / attack_total, 4) if attack_total > 0 else None
    frr = round(genuine_rejected_or_held / genuine_total, 4) if genuine_total > 0 else None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(records),
        "genuine_total": genuine_total,
        "attack_total": attack_total,
        "inconclusive_count": inconclusive_count,
        "far": far,
        "frr": frr,
        "confusion_matrix": {
            "genuine_accepted_tp": genuine_accepted,
            "genuine_rejected_fn": genuine_rejected_or_held,
            "attack_blocked_tn": attack_blocked,
            "attack_accepted_fp": attack_accepted,
        },
        "by_category": dict(by_category),
        "by_accent": dict(by_accent),
    }


def _discover_dataset(dataset_dir: Path) -> list[tuple[str, str, Path]]:
    """Find labeled .wav files arranged under genuine/, noisy/, cloned/, impostor/."""
    discovered: list[tuple[str, str, Path]] = []
    for label in sorted(ALL_LABELS):
        root = dataset_dir / label
        if not root.is_dir():
            continue
        for wav_path in sorted(root.rglob("*.wav")):
            rel_parts = wav_path.relative_to(root).parts
            accent = rel_parts[0] if len(rel_parts) > 1 else "unspecified"
            discovered.append((label, accent, wav_path))
    return discovered


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate VoiceSuraksha-Sync accuracy against DetectifAI API.")
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Folder containing genuine/, noisy/, cloned/, impostor/ WAVs.")
    parser.add_argument("--speaker-id", required=True, help="Enrolled provider speaker_id in DetectifAI Eidos.")
    parser.add_argument("--output", type=Path, default=Path("reports/eval_accuracy.json"), help="Output JSON report path.")
    args = parser.parse_args()

    items = _discover_dataset(args.dataset_dir)
    if not items:
        print(f"No labeled WAV files found in {args.dataset_dir} under {sorted(ALL_LABELS)}.", file=sys.stderr)
        return 1

    client = DetectifClient()
    records: list[dict[str, Any]] = []
    for label, accent, wav_path in items:
        wav_bytes = wav_path.read_bytes()
        try:
            vres = client.verify_with_deepfake(speaker_id=args.speaker_id, wav_bytes=wav_bytes)
            voice_in = VoiceEvidenceInput(
                speaker_decision=vres.speaker_decision,
                deepfake_verdict=vres.deepfake_verdict,
                duration_seconds=vres.duration_seconds,
                speaker_similarity=vres.speaker_similarity,
                speaker_threshold=vres.speaker_threshold,
            )
        except DetectifProviderError as exc:
            voice_in = VoiceEvidenceInput(provider_error=exc.error_code)

        decision = evaluate_voice_authorization(
            session=SessionPolicyInput(
                sector="telecom",
                action="sim_replacement",
                resource_ref=f"EVAL-{wav_path.stem}",
                require_phrase_match=False,
            ),
            voice_result=voice_in,
            phrase_match=True,
            active_risk_events=[],
        )
        records.append({
            "file": str(wav_path),
            "label": label,
            "accent": accent,
            "outcome": decision.outcome,
            "reason_codes": decision.reason_codes,
        })

    report = evaluate_labeled_records(records)
    report["speaker_id"] = args.speaker_id
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
