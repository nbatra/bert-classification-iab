"""
Background script to correct labels and generate keywords for all domains using Sonnet 4.

For each domain, Sonnet returns:
- The single best IAB Tier-1 category (correcting Kaggle's noisy labels)
- 5-10 English keywords describing the website content

Usage:
    .venv/bin/python scripts/run_sonnet_correction.py

Resumes automatically from checkpoints if interrupted.
"""

import json
import os
import subprocess
import time
import warnings
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
CORRECTED_DIR = DATA_DIR / 'corrected'
CORRECTED_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR = CORRECTED_DIR / 'sonnet_checkpoints'
CHECKPOINT_DIR.mkdir(exist_ok=True)

CONFIG_PATH = PROJECT_DIR / 'config.json'
if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"Missing {CONFIG_PATH}. Copy config.json.template to config.json and fill in your AWS account ID."
    )
with open(CONFIG_PATH) as f:
    _config = json.load(f)

SONNET_MODEL_ID = _config['sonnet_model_id']
MAX_WORKERS = _config.get('max_workers_sonnet', 10)
CHECKPOINT_EVERY = _config.get('checkpoint_every', 200)
MAX_RETRIES = 3


def get_bedrock_client():
    result = subprocess.run(
        ['aws', 'configure', 'export-credentials', '--profile', 'default', '--format', 'process'],
        capture_output=True, text=True,
        env={**dict(os.environ), 'AWS_CA_BUNDLE': ''}
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to export credentials: {result.stderr}")
    creds = json.loads(result.stdout)
    return AnthropicBedrock(
        aws_access_key=creds['AccessKeyId'],
        aws_secret_key=creds['SecretAccessKey'],
        aws_session_token=creds['SessionToken'],
        aws_region='us-east-1',
        http_client=httpx.Client(verify=False, timeout=120.0),
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


# Load categories
train_df = pd.read_parquet(PROCESSED_DIR / 'train.parquet')
label_info = pd.read_parquet(PROCESSED_DIR / 'label_info.parquet')
CATEGORIES = sorted(label_info['tier1'].unique().tolist())

SYSTEM_PROMPT = f"""You are a website classification expert. Given a domain and its metadata, return:
1. The single most appropriate IAB Tier-1 category
2. 5-10 English keywords describing the website content

Valid categories: {json.dumps(CATEGORIES)}

Rules:
- Pick exactly ONE category (the best fit)
- Keywords must be in English regardless of the website language
- Keywords should describe the actual content/purpose of the site
- Return ONLY valid JSON in this exact format:
{{"category": "<category_name>", "keywords": "keyword1, keyword2, keyword3, ..."}}"""


def classify_domain(client, domain, title=None, description=None):
    parts = [f"Domain: {domain}"]
    if title and len(str(title).strip()) > 2:
        parts.append(f"Title: {str(title).strip()[:200]}")
    if description and len(str(description).strip()) > 2:
        parts.append(f"Description: {str(description).strip()[:300]}")

    msg = client.messages.create(
        model=SONNET_MODEL_ID,
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=[{'role': 'user', 'content': '\n'.join(parts)}]
    )
    response_text = msg.content[0].text.strip()
    if response_text.startswith('```'):
        response_text = response_text.split('```')[1]
        if response_text.startswith('json'):
            response_text = response_text[4:]
    result = json.loads(response_text)
    return result, msg.usage.input_tokens, msg.usage.output_tokens


def classify_with_retry(client, domain, title, description):
    for attempt in range(MAX_RETRIES):
        try:
            result, in_tok, out_tok = classify_domain(client, domain, title, description)
            # Validate category
            if result.get('category') not in CATEGORIES:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(1.0)
                    continue
                result['category'] = '_INVALID'
            return result, in_tok, out_tok
        except (json.JSONDecodeError, KeyError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1.0)
            else:
                return {'category': '_PARSE_ERROR', 'keywords': ''}, 0, 0
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2.0 * (2 ** attempt))
            else:
                return {'category': '_ERROR', 'keywords': '', '_message': str(e)[:200]}, 0, 0


def main():
    print("=" * 60)
    print("DATA CORRECTION: SONNET 4 (ALL DOMAINS)")
    print("=" * 60)

    # Load all unique domains across train/val/test
    val_df = pd.read_parquet(PROCESSED_DIR / 'val.parquet')
    test_df = pd.read_parquet(PROCESSED_DIR / 'test.parquet')
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    all_df['_text_len'] = all_df['title'].fillna('').str.len() + all_df['description'].fillna('').str.len()
    all_domains = all_df.sort_values('_text_len', ascending=False).drop_duplicates(
        subset='domain_clean', keep='first'
    ).drop(columns=['_text_len']).reset_index(drop=True)

    existing = load_existing_results()
    valid_existing = {k: v for k, v in existing.items() if not v.get('category', '').startswith('_')}
    errors_existing = len(existing) - len(valid_existing)
    print(f"Existing checkpoints: {len(existing):,} domains ({len(valid_existing):,} valid, {errors_existing:,} errors to retry)")
    print(f"Total domains: {len(all_domains):,}")

    # Filter to unlabeled -- only skip domains with valid results, retry errors
    remaining = all_domains[~all_domains['domain_clean'].isin(valid_existing.keys())].reset_index(drop=True)
    print(f"Remaining to label: {len(remaining):,}")

    if len(remaining) == 0:
        print("All domains already labeled.")
        return

    client = get_bedrock_client()

    # Verify connection
    test_msg = client.messages.create(
        model=SONNET_MODEL_ID, max_tokens=10,
        messages=[{'role': 'user', 'content': 'Say OK'}]
    )
    print(f"\nSonnet 4 connection verified: {test_msg.content[0].text}")
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

    rows = remaining.to_dict('records')

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_row, row): row['domain_clean'] for row in rows}

        for future in as_completed(futures):
            try:
                domain, (result, in_tok, out_tok) = future.result()
                batch_results[domain] = result
                total_input += in_tok
                total_output += out_tok

                if result.get('category', '').startswith('_'):
                    errors += 1

                completed += 1

                if len(batch_results) >= CHECKPOINT_EVERY:
                    save_checkpoint(batch_results, batch_num)
                    batch_num += 1
                    batch_results = {}

                if completed % 1000 == 0:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed
                    eta = (len(remaining) - completed) / rate if rate > 0 else 0
                    print(f"  [{completed:,}/{len(remaining):,}] {rate:.1f}/s | "
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

    all_results = load_existing_results()
    print(f"\nTotal labeled domains (all checkpoints): {len(all_results):,}")


if __name__ == '__main__':
    main()
