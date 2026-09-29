"""Average two self-trained, compatible student checkpoints.

Select the mixture on validation only; retain both checkpoint ancestors and logs.
"""
import argparse
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--second-weight', type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.second_weight <= 1:
        parser.error('--second-weight must be between 0 and 1')
    a = torch.load(args.first, map_location='cpu', weights_only=True)
    b = torch.load(args.second, map_location='cpu', weights_only=True)
    for key in ('protocol', 'implementation', 'config'):
        if a[key] != b[key]:
            raise ValueError(f'Checkpoints differ in {key}')
    if a['model'].keys() != b['model'].keys():
        raise ValueError('State dictionaries have different keys')
    mixed = {}
    w = args.second_weight
    for key, x in a['model'].items():
        y = b['model'][key]
        if x.shape != y.shape or x.dtype != y.dtype:
            raise ValueError(f'Incompatible tensor: {key}')
        mixed[key] = (x.float().mul(1-w).add_(y.float(), alpha=w)).to(x.dtype) if x.is_floating_point() else x
    result = dict(b)
    result['model'] = mixed
    result['train_tokens'] = a['train_tokens'] + b['train_tokens']
    result['averaging'] = {'first': str(args.first), 'second': str(args.second), 'second_weight': w,
                           'ancestor_train_tokens': [a['train_tokens'], b['train_tokens']]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.save(result, args.output)
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
