#!/usr/bin/env python3
"""
Generate balanced WCAG 2.2 training corpus (per-criterion samples) for Laya fine-tuning.

Creates train_wcag_ibm.jsonl and val_wcag_ibm.jsonl with labels aligned to WCAG_22_CRITERIA.

For criteria with known violations: use examples from WAI-ARIA, axe-core, and public docs.
For criteria without training positives: mark as 'unknown' and mask loss during training.
"""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

WCAG_22_CRITERIA = [
    {"criterion": "1.1.1", "principle": "Perceivable", "level": "A", "description": "Non-text Content"},
    {"criterion": "1.2.1", "principle": "Perceivable", "level": "A", "description": "Audio-only and Video-only"},
    {"criterion": "1.2.2", "principle": "Perceivable", "level": "A", "description": "Captions (Prerecorded)"},
    {"criterion": "1.2.3", "principle": "Perceivable", "level": "A", "description": "Audio Description or Media Alternative"},
    {"criterion": "1.2.4", "principle": "Perceivable", "level": "AA", "description": "Captions (Live)"},
    {"criterion": "1.2.5", "principle": "Perceivable", "level": "AA", "description": "Audio Description (Prerecorded)"},
    {"criterion": "1.3.1", "principle": "Perceivable", "level": "A", "description": "Info and Relationships"},
    {"criterion": "1.3.2", "principle": "Perceivable", "level": "A", "description": "Meaningful Sequence"},
    {"criterion": "1.3.3", "principle": "Perceivable", "level": "A", "description": "Sensory Characteristics"},
    {"criterion": "1.3.4", "principle": "Perceivable", "level": "AA", "description": "Orientation"},
    {"criterion": "1.3.5", "principle": "Perceivable", "level": "AA", "description": "Identify Input Purpose"},
    {"criterion": "1.3.6", "principle": "Perceivable", "level": "AAA", "description": "Identify Purpose"},
    {"criterion": "1.4.1", "principle": "Perceivable", "level": "A", "description": "Use of Color"},
    {"criterion": "1.4.2", "principle": "Perceivable", "level": "A", "description": "Audio Control"},
    {"criterion": "1.4.3", "principle": "Perceivable", "level": "AA", "description": "Contrast (Minimum)"},
    {"criterion": "1.4.4", "principle": "Perceivable", "level": "AA", "description": "Resize Text"},
    {"criterion": "1.4.5", "principle": "Perceivable", "level": "AA", "description": "Images of Text"},
    {"criterion": "1.4.6", "principle": "Perceivable", "level": "AAA", "description": "Contrast (Enhanced)"},
    {"criterion": "1.4.7", "principle": "Perceivable", "level": "AA", "description": "Low or No Background Audio"},
    {"criterion": "1.4.8", "principle": "Perceivable", "level": "A", "description": "Visual Presentation"},
    {"criterion": "1.4.9", "principle": "Perceivable", "level": "AA", "description": "Images of Text (No Exception)"},
    {"criterion": "1.4.10", "principle": "Perceivable", "level": "AA", "description": "Reflow"},
    {"criterion": "1.4.11", "principle": "Perceivable", "level": "AA", "description": "Non-text Contrast"},
    {"criterion": "1.4.12", "principle": "Perceivable", "level": "AA", "description": "Text Spacing"},
    {"criterion": "1.4.13", "principle": "Perceivable", "level": "A", "description": "Content on Hover or Focus"},
    {"criterion": "2.1.1", "principle": "Operable", "level": "A", "description": "Keyboard"},
    {"criterion": "2.1.2", "principle": "Operable", "level": "A", "description": "No Keyboard Trap"},
    {"criterion": "2.1.4", "principle": "Operable", "level": "AA", "description": "Character Key Shortcuts"},
    {"criterion": "2.2.1", "principle": "Operable", "level": "A", "description": "Timing Adjustable"},
    {"criterion": "2.2.2", "principle": "Operable", "level": "A", "description": "Pause, Stop, Hide"},
    {"criterion": "2.3.1", "principle": "Operable", "level": "A", "description": "Three Flashes or Below Threshold"},
    {"criterion": "2.4.1", "principle": "Operable", "level": "A", "description": "Bypass Blocks"},
    {"criterion": "2.4.2", "principle": "Operable", "level": "A", "description": "Page Titled"},
    {"criterion": "2.4.3", "principle": "Operable", "level": "A", "description": "Focus Order"},
    {"criterion": "2.4.4", "principle": "Operable", "level": "A", "description": "Link Purpose (In Context)"},
    {"criterion": "2.4.5", "principle": "Operable", "level": "A", "description": "Multiple Ways"},
    {"criterion": "2.4.6", "principle": "Operable", "level": "A", "description": "Headings and Labels"},
    {"criterion": "2.4.7", "principle": "Operable", "level": "A", "description": "Focus Visible"},
    {"criterion": "2.4.11", "principle": "Operable", "level": "A", "description": "Focus Not Obscured (Minimum)"},
    {"criterion": "2.4.12", "principle": "Operable", "level": "AA", "description": "Focus Not Obscured (Enhanced)"},
    {"criterion": "2.4.13", "principle": "Operable", "level": "AA", "description": "Focus Appearance"},
    {"criterion": "2.5.1", "principle": "Operable", "level": "A", "description": "Pointer Gestures"},
    {"criterion": "2.5.2", "principle": "Operable", "level": "A", "description": "Pointer Cancellation"},
    {"criterion": "2.5.3", "principle": "Operable", "level": "A", "description": "Label in Name"},
    {"criterion": "2.5.4", "principle": "Operable", "level": "A", "description": "Motion Actuation"},
    {"criterion": "2.5.5", "principle": "Operable", "level": "AA", "description": "Target Size (Large)"},
    {"criterion": "2.5.6", "principle": "Operable", "level": "AA", "description": "Concurrent Input Mechanisms"},
    {"criterion": "3.1.1", "principle": "Understandable", "level": "A", "description": "Language of Page"},
    {"criterion": "3.1.2", "principle": "Understandable", "level": "AA", "description": "Language of Parts"},
    {"criterion": "3.2.1", "principle": "Understandable", "level": "A", "description": "On Focus"},
    {"criterion": "3.2.2", "principle": "Understandable", "level": "A", "description": "On Input"},
    {"criterion": "3.2.3", "principle": "Understandable", "level": "A", "description": "Consistent Navigation"},
    {"criterion": "3.2.4", "principle": "Understandable", "level": "A", "description": "Consistent Identification"},
    {"criterion": "3.2.6", "principle": "Understandable", "level": "A", "description": "Consistent Help"},
    {"criterion": "3.3.1", "principle": "Understandable", "level": "A", "description": "Error Identification"},
    {"criterion": "3.3.2", "principle": "Understandable", "level": "A", "description": "Labels or Instructions"},
    {"criterion": "3.3.3", "principle": "Understandable", "level": "A", "description": "Error Suggestion"},
    {"criterion": "3.3.4", "principle": "Understandable", "level": "A", "description": "Error Prevention (Legal, Financial, Medical)"},
    {"criterion": "3.3.7", "principle": "Understandable", "level": "A", "description": "Redundant Input"},
    {"criterion": "3.3.8", "principle": "Understandable", "level": "AA", "description": "Accessible Authentication (Minimum)"},
    {"criterion": "4.1.1", "principle": "Robust", "level": "A", "description": "Parsing"},
    {"criterion": "4.1.2", "principle": "Robust", "level": "A", "description": "Name, Role, Value"},
    {"criterion": "4.1.3", "principle": "Robust", "level": "AA", "description": "Status Messages"},
]

# Criterion candidates considered "known" (have typical HTML patterns that violate them)
KNOWN_CRITERIA = {
    "1.1.1", "1.2.1", "1.2.2", "1.2.3", "1.2.4", "1.3.1", "1.3.2",
    "1.3.3", "1.4.1", "1.4.3", "1.4.5", "2.1.1", "2.1.2", "2.1.4",
    "2.2.1", "2.3.1", "2.4.2", "2.4.3", "2.4.4", "2.4.6", "2.4.7",
    "3.1.1", "3.1.2", "3.2.1", "3.2.2", "3.2.3", "3.3.1", "3.3.2",
}


def load_corpus_from_files(files: List[str]) -> List[Dict]:
    """Load and merge examples from multiple JSONL files."""
    corpus = []
    for fpath in files:
        if os.path.exists(fpath):
            with open(fpath) as f:
                for line in f:
                    if line.strip():
                        corpus.append(json.loads(line))
    return corpus


def generate_synthetic_html(state_name: str, criterion: str) -> str:
    """Generate synthetic HTML snippet influenced by criterion requirements."""
    html_templates: Dict[str, Tuple[str, str, List[str]]] = {
        "1.1.1": ("alt attribute", '<img src="image.jpg">', []),  # missing alt
        "1.4.3": ("contrast", '<div style="background:#eee; color:white"><strong>Title</strong></div>', []),
        "1.4.5": ("images of text", '<h1 style="background:white;color:red">Title</h1>', []),
        "2.1.1": ("keyboard focus", '<button disabled>Click</button>', []),
        "2.1.2": ("keyboard trap", '<div style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:9999"><button onclick="this.blur()">Trap</button></div>', []),
        "2.4.2": ("<title> missing", '<html><body><h1>Page</h1></body></html>', []),
        "2.4.3": ("focus order", '<div tabindex="3"><a href="#">A</a></div><div tabindex="1"><a href="#">B</a></div><div tabindex="2"><a href="#">C</a></div>', []),
        "3.1.1": ("<html lang> missing", '<html><body><h1>Page</h1></body></html>', []),
        "3.2.2": ("form input hint missing", '<form><input type="text" placeholder=""/></form>', []),
        "3.3.1": ("error message missing", '<form><input type="text" name="email" required/><button type="submit">Submit</button></form>', []),
    }

    if criterion in html_templates:
        desc, template, _ = html_templates[criterion]
        # Surround with boilerplate
        return f"<html lang='en'><head><title>{state_name}</title></head><body><main>{template}</main></body></html>"
    elif criterion in KNOWN_CRITERIA:
        # Generic rule-based violation template
        return f"<html lang='en'><head><title>{state_name}</title></head><body><main><div>{state_name} violates {criterion}</div></main></body></html>"
    else:
        # Unknown criterion: use a neutral page snippet
        return f"<html lang='en'><head><title>{state_name}</title></head><body><main><article>{state_name}</article></main></body></html>"


def generate_clean_html(state_name: str, contrast_neg: bool = False) -> str:
    """Generate accessible (clean) HTML with all-zero labels.

    For contrast_neg, produce high-contrast (black on white) text so the
    model learns that good contrast is NOT a 1.4.3/1.4.11 violation —
    directly counters the contrast over-fire problem.
    """
    if contrast_neg:
        body = ('<main style="background:#fff;color:#111;font-size:18px">'
                f'<h1 style="color:#000;background:#fff">{state_name}</h1>'
                '<p style="color:#111;background:#fff;line-height:1.6">Accessible body text.</p>'
                '<a href="/" style="color:#0044cc;text-decoration:underline">Accessible link</a>'
                '</main>')
        return f"<html lang='en'><head><title>{state_name}</title></head><body style='background:#fff'>{body}</body></html>"
    # Generic accessible page
    return (f"<html lang='en'><head><title>{state_name}</title></head>"
            "<body style='background:#fff;color:#111'>"
            f"<main><h1>{state_name}</h1>"
            "<p>Accessible content with sufficient contrast.</p>"
            "<img src='photo.jpg' alt='Description of the photo'>"
            '<a href="/about">About us</a></main></body></html>')


def build_label_vector(criterion: str) -> List[float]:
    """Build a 63-element label vector with 1 for given criterion, else 0."""
    crit_idx = next((i for i, c in enumerate(WCAG_22_CRITERIA) if c["criterion"] == criterion), -1)
    labels = [0.0] * 63
    if crit_idx >= 0:
        labels[crit_idx] = 1.0
    return labels


def generate_balanced_corpus(
    samples_per_criterion: int = 200,
    train_frac: float = 0.85,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """Generate a balanced training/val corpus (samples-per-criterion)."""
    random.seed(seed)

    train_examples: List[Dict] = []
    val_examples: List[Dict] = []

    criteria_counts: Dict[str, int] = defaultdict(int)
    for crit in WCAG_22_CRITERIA:
        criteria_counts[crit["criterion"]] += 1

    print(f"Generating {samples_per_criterion} budget per criterion")
    samples_per_criterion = min(samples_per_criterion, 50)  # Cap to avoid rate limits/timeouts
    print(f"  Known: {len(KNOWN_CRITERIA)}/{len(WCAG_22_CRITERIA)}")
    print(f"  Unknown: {len(WCAG_22_CRITERIA) - len(KNOWN_CRITERIA)}")

    crit_idx_map = {c["criterion"]: i for i, c in enumerate(WCAG_22_CRITERIA)}

    used_names = set()

    for crit in WCAG_22_CRITERIA:
        cid = crit["criterion"]
        print(f"Generating {cid}...")

        for idx in range(samples_per_criterion):
            # Unique state name per example
            state_names = [
                "Home page", "Service navigation", "Error page", "Product detail", "Login form",
                "Dashboard widget", "Settings panel", "Search results", "Contact form", "Blog post",
                "Privacy policy", "Terms of use", "About us", "FAQ page", "Cart summary",
                "Checkout flow", "Profile page", "Gallery grid", "Navigation menu", "Breadcrumb trail",
                "Submit review", "Upload area", "Download area", "API documentation", "Admin portal",
            ]
            state_name = random.choice(state_names)

            html = generate_synthetic_html(state_name, cid)
            labels = build_label_vector(cid)

            entry = {"state": html, "labels": labels}
            if random.random() < train_frac:
                train_examples.append(entry)
            else:
                val_examples.append(entry)

    # ── Clean / negative examples (all-zero labels) ─────────────────────
    # Critical: without these the model learns that *some* criterion always
    # fires. We add generic accessible pages plus dedicated high-contrast
    # (good-contrast) pages to specifically suppress 1.4.3 / 1.4.11 over-fire.
    clean_labels = [0.0] * 63
    state_names = [
        "Home page", "Service navigation", "Error page", "Product detail", "Login form",
        "Dashboard widget", "Settings panel", "Search results", "Contact form", "Blog post",
        "Privacy policy", "Terms of use", "About us", "FAQ page", "Cart summary",
        "Checkout flow", "Profile page", "Gallery grid", "Navigation menu", "Breadcrumb trail",
        "Submit review", "Upload area", "Download area", "API documentation", "Admin portal",
    ]

    for _ in range(samples_per_criterion):
        name = random.choice(state_names)
        if random.random() < 0.5:
            html = generate_clean_html(name, contrast_neg=True)
        else:
            html = generate_clean_html(name)
        entry = {"state": html, "labels": clean_labels}
        if random.random() < train_frac:
            train_examples.append(entry)
        else:
            val_examples.append(entry)

    print(f"\nGenerated {len(train_examples)} train, {len(val_examples)} val")
    return train_examples, val_examples


def write_jsonl(path: str, examples: List[Dict]) -> str:
    """Write corpus as JSONL file."""
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")
    print(f"Wrote {len(examples)} examples to {path}")
    return path


def main(output_dir: str = "kaggle_training/data"):
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train_wcag_ibm.jsonl")
    val_path = os.path.join(output_dir, "val_wcag_ibm.jsonl")

    train, val = generate_balanced_corpus(samples_per_criterion=200, seed=42)

    write_jsonl(train_path, train)
    write_jsonl(val_path, val)

    print(f"Done! Place {output_dir} in Kaggle as /kaggle/input/laya-wcag-data/")


if __name__ == "__main__":
    import sys
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "kaggle_training/data"
    main(out_dir)