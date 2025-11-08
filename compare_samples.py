#!/usr/bin/env python3
import json

print('=== Working seed110 (first few samples) ===')
try:
    with open('samples/config.ema_0.9999_000500.pt.seed110_step0.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 3: break
            sample = json.loads(line)
            print(f'Q: {sample["question"]}')
            print(f'A: "{sample["generate_answer"]}"')
            print(f'Conf: {sample["confidence"]}')
            print()
except FileNotFoundError:
    print("seed110 file not found")

print('=== Non-working seed119 (first few samples) ===')
try:
    with open('samples/config.ema_0.9999_000500.pt.seed119_step0.jsonl', 'r') as f:
        for i, line in enumerate(f):
            if i >= 3: break
            sample = json.loads(line)
            print(f'Q: {sample["question"]}')
            print(f'A: "{sample["generate_answer"]}"')
            print(f'Conf: {sample["confidence"]}')
            print()
except FileNotFoundError:
    print("seed119 file not found")

# Let's also check what keys are available in both files
print('=== Available keys comparison ===')
try:
    with open('samples/config.ema_0.9999_000500.pt.seed110_step0.jsonl', 'r') as f:
        sample110 = json.loads(next(f))
        print(f'seed110 keys: {list(sample110.keys())}')
except:
    print("Could not read seed110")

try:
    with open('samples/config.ema_0.9999_000500.pt.seed119_step0.jsonl', 'r') as f:
        sample119 = json.loads(next(f))
        print(f'seed119 keys: {list(sample119.keys())}')
except:
    print("Could not read seed119")