"""
Background script to label domains with Claude Opus 4.6 soft probability distributions.

Strategy:
- Select domains from ALL splits (train/val/test), stratified by Tier-1 category
- Within each category, prioritize domains with the richest text (longest title+description)
- Skip domains already labeled with valid results in checkpoints (retry errors)
- Save progress every 200 domains

Usage:
    .venv/bin/python scripts/run_teacher_labeling.py

Resumes automatically from checkpoints if interrupted.
"""

import json
import os
import subprocess
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import urllib3

urllib3.disable_warnings()
warnings.filterwarnings('ignore')

os.environ.pop('AWS_BEARER_TOKEN_BEDROCK', None)
os.environ.pop('ANTHROPIC_API_KEY', None)

import httpx
from anthropic import AnthropicBedrock

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
CHECKPOINT_DIR = PROCESSED_DIR / 'teacher_checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

CONFIG_PATH = PROJECT_DIR / 'config.json'
if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Missing {CONFIG_PATH}. Copy config.json.template to config.json and fill in your AWS account ID."
    )
with open(CONFIG_PATH) as f:
    _config = json.load(f)

MODEL_ID = _config['opus_model_id']
MAX_WORKERS = _config.get('max_workers_opus', 5)
CHECKPOINT_EVERY = _config.get('checkpoint_every', 200)
MAX_RETRIES = 3
TARGET_DOMAINS = 10000  # per run; accumulates across restarts


def get_bedrock_client():
    result = subprocess.run(
        ['aws', 'configure', 'export-credentials', '--profile', 'default', '--format', 'process'],
        capture_output=True, text=True,
        env={**dict(os.environ), 'AWS_CA_BUNDLE': ''}
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to export credentials: {result.stderr}")
    creds = json.loads(result.stdout)
    http_client = httpx.Client(verify=False, timeout=120.0)
    return AnthropicBedrock(
        aws_access_key=creds['AccessKeyId'],
        aws_secret_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken'],
        aws_region='us-east-1',
        http_client=http_client,
    )


def load_existing_results():
    results = {}
    for f in sorted(CHECKPOINT_DIR.glob('checkpoint_*.json')):
        with open(f) as fh:
            results.update(json.load(fh))
    return results


def save_checkpoint(results, batch_num):
    path = CHECKPOINT_DIR / f'checkpoint_{batch_num:04d}.json'
    with open(path, 'w') as f:
        json.dump(results, f)
    print(f"  [checkpoint] Saved {len(results)} domains to {path.name}")


def select_domains(all_domains_df, existing_results, target=TARGET_DOMAINS):
    """Select domains stratified by tier1, prioritizing text-rich domains.
    Uses all splits (train/val/test). Only skips domains with valid results."""
    domains = all_domains_df.drop_duplicates(subset='domain_clean')[
        ['domain_clean', 'tier1', 'text', 'title', 'description']
    ].copy()
    domains['text_richness'] = domains['text'].str.len()

    # Only skip domains with valid results (retry errors)
    valid_existing = {k for k, v in existing_results.items()
                     if '_ERROR' not in v and '_PARSE_ERROR' not in v and '_INVALID_CATEGORIES' not in v}
    domains = domains[~domains['domain_clean'].isin(valid_existing)]

    if len(domains) == 0:
        return pd.DataFrame()

    # Allocate per category proportionally with floor of 50
    cat_counts = domains['tier1'].value_counts()
    actual_target = min(target, len(domains))
    proportional = (cat_counts / cat_counts.sum() * actual_target).astype(int).clip(lower=min(50, cat_counts.min()))
    scale = actual_target / proportional.sum()
    allocation = (proportional * scale).astype(int)

    # Adjust to hit target
    diff = actual_target - allocation.sum()
    if diff > 0:
        for cat in allocation.sort_values(ascending=False).index[:diff]:
            allocation[cat] += 1
    elif diff < 0:
        for cat in allocation.sort_values(ascending=True).index[:abs(diff)]:
            allocation[cat] = max(1, allocation[cat] - 1)

    # Sample: pick richest-text domains per category
    selected = []
    for cat, n in allocation.items():
        cat_domains = domains[domains['tier1'] == cat].nlargest(n, 'text_richness')
        selected.append(cat_domains)

    result = pd.concat(selected, ignore_index=True)
    return result


# Load all splits for domain selection
train_df = pd.read_parquet(PROCESSED_DIR / 'train.parquet')
val_df = pd.read_parquet(PROCESSED_DIR / 'val.parquet')
test_df = pd.read_parquet(PROCESSED_DIR / 'test.parquet')
all_domains_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
TIER1_CATEGORIES = sorted(train_df['tier1'].unique().tolist())

SYSTEM_PROMPT = f"""You are a website classification expert. Your task is to classify website domains into IAB Content Taxonomy Tier-1 categories.

The valid categories are:
{json.dumps(TIER1_CATEGORIES, indent=2)}

Rules:
1. Return a JSON object with category names as keys and confidence scores (0.0 to 1.0) as values.
2. Only include categories with confidence >= 0.1.
3. At least one category must have confidence >= 0.5.
4. Confidence scores should be calibrated: 0.9+ means very certain, 0.5-0.9 means likely, 0.1-0.5 means plausible.
5. Return ONLY valid JSON. No explanations, no markdown, no extra text.
6. Use EXACTLY the category names listed above (case-sensitive).
"""


def build_user_prompt(domain, title=None, description=None):
    parts = [f"Domain: {domain}"]
    if title and len(str(title).strip()) > 2:
        parts.append(f"Title: {str(title).strip()[:200]}")
    if description and len(str(description).strip()) > 2:
        parts.append(f"Description: {str(description).strip()[:300]}")
    return '\n'.join(parts)


def classify_domain(client, domain, title=None, description=None):
    user_prompt = build_user_prompt(domain, title, description)
    message = client.messages.create(
        model=MODEL_ID,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': user_prompt}]
    )
    response_text = message.content[0].text.strip()
    if response_text.startswith('```'):
        response_text = response_text.split('```')[1]
        if response_text.startswith('json'):
            response_text = response_text[4:]
    try:
        scores = json.loads(response_text)
    except json.JSONDecodeError:
        scores = {'_PARSE_ERROR': 1.0, '_raw': response_text}
    return {
        'scores': scores,
        'input_tokens': message.usage.input_tokens,
        'output_tokens': message.usage.output_tokens,
    }


def classify_with_retry(client, domain, title, description):
    for attempt in range(MAX_RETRIES):
        try:
            result = classify_domain(client, domain, title, description)
            if '_PARSE_ERROR' not in result['scores']:
                invalid_cats = [c for c in result['scores'] if c not in TIER1_CATEGORIES]
                if invalid_cats:
                    result['scores'] = {k: v for k, v in result['scores'].items() if k in TIER1_CATEGORIES}
                    if not result['scores']:
                        result['scores'] = {'_INVALID_CATEGORIES': 1.0}
            return result
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2.0 * (2 ** attempt))
            else:
                return {'scores': {'_ERROR': 1.0, '_message': str(e)[:200]}, 'input_tokens': 0, 'output_tokens': 0}


def main():
    print("=" * 60)
    print("TEACHER LABELING: STRATIFIED 10K DOMAINS")
    print("=" * 60)

    existing = load_existing_results()
    valid_count = sum(1 for v in existing.values()
                     if '_ERROR' not in v and '_PARSE_ERROR' not in v and '_INVALID_CATEGORIES' not in v)
    print(f"Existing checkpoints: {len(existing):,} domains ({valid_count:,} valid)")

    selected = select_domains(all_domains_df, existing, TARGET_DOMAINS)
    print(f"Selected for labeling: {len(selected):,} domains")
    print(f"Category distribution:")
    for cat, count in selected['tier1'].value_counts().head(10).items():
        print(f"  {cat:30s}: {count}")
    print(f"  ... ({selected['tier1'].nunique()} categories total)")

    if len(selected) == 0:
        print("Nothing to label -- all target domains already in checkpoints.")
        return

    client = get_bedrock_client()

    # Verify connection
    test_msg = client.messages.create(
        model=MODEL_ID, max_tokens=10,
        messages=[{'role': 'user', 'content': 'Say OK'}]
    )
    print(f"\nBedrock connection verified: {test_msg.content[0].text}")
    print(f"Starting labeling with {MAX_WORKERS} workers...\n")

    batch_results = {}
    batch_num = len(list(CHECKPOINT_DIR.glob('checkpoint_*.json')))
    start_time = time.time()
    completed = 0
    errors = 0
    total_input = 0
    total_output = 0

    def process_row(row):
        return row['domain_clean'], classify_with_retry(
            client, row['domain_clean'],
            row['title'] if pd.notna(row['title']) else None,
            row['description'] if pd.notna(row['description']) else None
        )

    rows = selected.to_dict('records')

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row): row['domain_clean'] for row in rows}

        for future in as_completed(futures):
            try:
                domain, result = future.result()
                batch_results[domain] = result['scores']
                total_input += result['input_tokens']
                total_output += result['output_tokens']

                if '_ERROR' in result['scores'] or '_PARSE_ERROR' in result['scores']:
                    errors += 1

                completed += 1

                if len(batch_results) >= CHECKPOINT_EVERY:
                    save_checkpoint(batch_results, batch_num)
                    batch_num += 1
                    batch_results = {}

                if completed % 500 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(selected) - completed) / rate if rate > 0 else 0
                    print(f"  [{completed:,}/{len(selected):,}] {rate:.1f}/s | "
                          f"ETA: {eta/60:.0f}min | Errors: {errors} | "
                          f"Tokens: {total_input:,}in/{total_output:,}out")
            except Exception as e:
                errors += 1
                completed += 1

    # Final checkpoint
    if batch_results:
        save_checkpoint(batch_results, batch_num)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")
    print(f"Labeled: {completed:,} domains in {elapsed/60:.1f} minutes")
    print(f"Rate: {completed/elapsed:.1f} domains/sec")
    print(f"Errors: {errors} ({errors/max(completed,1)*100:.1f}%)")
    print(f"Tokens: {total_input:,} input, {total_output:,} output")

    # Report final totals
    all_results = load_existing_results()
    print(f"\nTotal labeled domains (all checkpoints): {len(all_results):,}")


if __name__ == '__main__':
    main()
