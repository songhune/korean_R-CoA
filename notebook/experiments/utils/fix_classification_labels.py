#!/usr/bin/env python3
"""
Fix Unicode normalization issues in classification labels

Issues found:
1. '論' (U+F941) → normalize to '論' (U+8AD6) - 2 items
2. '禮義' (U+F9B6 U+7FA9) → normalize to '禮義' (U+79AE U+7FA9) - 6 items

These are Unicode Compatibility Ideographs that should be normalized
to their canonical forms for consistency.
"""

import json
import unicodedata
from pathlib import Path
from collections import Counter


def normalize_label(label: str) -> str:
    """
    Normalize Unicode Compatibility Ideographs to canonical forms

    Args:
        label: Original label string

    Returns:
        Normalized label string
    """
    # Explicit normalization for known issues
    normalized = label

    # Fix 論 variant (U+F941 → U+8AD6)
    if '\uF941' in normalized:
        normalized = normalized.replace('\uF941', '\u8AD6')

    # Fix 禮 variant (U+F9B6 → U+79AE)
    if '\uF9B6' in normalized:
        normalized = normalized.replace('\uF9B6', '\u79AE')

    # Also apply NFC normalization for other potential issues
    normalized = unicodedata.normalize('NFC', normalized)

    return normalized


def analyze_and_fix_classification(
    input_path: str,
    output_path: str = None,
    dry_run: bool = False
):
    """
    Analyze and fix Unicode issues in classification labels

    Args:
        input_path: Path to classification JSON file
        output_path: Output path (default: overwrite input)
        dry_run: If True, only show changes without writing
    """

    # Load data
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("=" * 80)
    print("Classification Label Unicode Normalization")
    print("=" * 80)

    # Analyze current state
    original_labels = Counter(item['label'] for item in data['data'])
    print(f"\nOriginal unique labels: {len(original_labels)}")

    # Track changes
    changes = []
    fixed_data = data.copy()

    for item in fixed_data['data']:
        original_label = item['label']
        normalized_label = normalize_label(original_label)

        if original_label != normalized_label:
            changes.append({
                'id': item['id'],
                'original': original_label,
                'original_unicode': ' '.join(f'U+{ord(c):04X}' for c in original_label),
                'normalized': normalized_label,
                'normalized_unicode': ' '.join(f'U+{ord(c):04X}' for c in normalized_label),
                'input_preview': item['input'][:50]
            })

            # Apply fix
            item['label'] = normalized_label

    # Analyze fixed state
    fixed_labels = Counter(item['label'] for item in fixed_data['data'])

    print(f"Fixed unique labels: {len(fixed_labels)}")
    print(f"\nChanges made: {len(changes)} items")

    if changes:
        print("\n" + "-" * 80)
        print("Items modified:")
        print("-" * 80)

        for change in changes:
            print(f"\nID: {change['id']}")
            print(f"  Original: '{change['original']}' ({change['original_unicode']})")
            print(f"  Fixed:    '{change['normalized']}' ({change['normalized_unicode']})")
            print(f"  Input:    {change['input_preview']}...")

    # Show label distribution changes
    print("\n" + "=" * 80)
    print("Label Distribution Changes:")
    print("=" * 80)

    # Identify affected labels
    affected_labels = set()
    for change in changes:
        affected_labels.add(change['original'])
        affected_labels.add(change['normalized'])

    print(f"\n{'Label':<15} {'Before':>10} {'After':>10} {'Change':>10}")
    print("-" * 50)

    for label in sorted(affected_labels):
        before = original_labels.get(label, 0)
        after = fixed_labels.get(label, 0)
        change_val = after - before
        change_str = f"+{change_val}" if change_val > 0 else str(change_val)
        print(f"{label:<15} {before:>10} {after:>10} {change_str:>10}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    print("=" * 80)
    print(f"Total items: {len(data['data'])}")
    print(f"Original unique labels: {len(original_labels)}")
    print(f"Fixed unique labels: {len(fixed_labels)}")
    print(f"Labels reduced by: {len(original_labels) - len(fixed_labels)}")
    print(f"Items modified: {len(changes)}")

    # Write output
    if not dry_run:
        output_path = output_path or input_path

        # Update metadata
        fixed_data['description'] = fixed_data['description']

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(fixed_data, f, ensure_ascii=False, indent=2)

        print(f"\n[SAVED] Fixed data written to: {output_path}")
    else:
        print("\n[DRY RUN] No changes written to disk")

    return fixed_data, changes


def verify_normalization(file_path: str):
    """
    Verify that all labels are properly normalized

    Args:
        file_path: Path to classification JSON file
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print("\n" + "=" * 80)
    print("VERIFICATION:")
    print("=" * 80)

    issues_found = []

    for item in data['data']:
        label = item['label']

        # Check for compatibility ideographs
        for char in label:
            if '\uF900' <= char <= '\uFAFF':  # CJK Compatibility Ideographs block
                issues_found.append({
                    'id': item['id'],
                    'label': label,
                    'char': char,
                    'unicode': f'U+{ord(char):04X}'
                })

    if issues_found:
        print(f"\n[WARNING] Found {len(issues_found)} items with compatibility ideographs:")
        for issue in issues_found:
            print(f"  ID {issue['id']}: '{issue['label']}' contains {issue['char']} ({issue['unicode']})")
        return False
    else:
        print("\n[OK] All labels are properly normalized")
        print("No Unicode Compatibility Ideographs found")
        return True


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Fix Unicode normalization issues in classification labels'
    )
    parser.add_argument('--input', type=str,
                       default='../../benchmark/kls_bench/kls_bench_classification.json',
                       help='Input classification JSON file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file (default: overwrite input)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show changes without writing')
    parser.add_argument('--verify', action='store_true',
                       help='Verify normalization after fix')

    args = parser.parse_args()

    # Run normalization
    fixed_data, changes = analyze_and_fix_classification(
        args.input,
        args.output,
        args.dry_run
    )

    # Verify if requested
    if args.verify and not args.dry_run:
        output_path = args.output or args.input
        verify_normalization(output_path)


if __name__ == '__main__':
    main()
