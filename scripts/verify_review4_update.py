#!/usr/bin/env python3
"""Verify a delivered v8 update and its exact v7 base without extracting files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile


def digest(stream):
    h = hashlib.sha256()
    size = 0
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        size += len(block)
        h.update(block)
    return size, h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--update', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    args = p.parse_args()
    with zipfile.ZipFile(args.update) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive member')
        manifest = json.loads(z.read('MANIFEST.json'))
        if manifest['schema'] != 'expose-v8-author-review-update-v1':
            raise ValueError('Not a v8 update manifest')
        if set(names) != set(manifest['files']) | {'MANIFEST.json'}:
            raise ValueError('Archive inventory differs from manifest')
        with args.base.open('rb') as f:
            base_size, base_hash = digest(f)
        base = manifest['base_archive']
        if base_size != base['bytes'] or base_hash != base['sha256']:
            raise ValueError('The supplied base is not the bound v7 archive')
        for name, expected in manifest['files'].items():
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
                raise ValueError('Unsafe archive path')
            if stat.S_ISLNK(z.getinfo(name).external_attr >> 16):
                raise ValueError('Archive symlink')
            with z.open(name) as f:
                size, sha = digest(f)
            if size != expected['bytes'] or sha != expected['sha256']:
                raise ValueError('Payload identity differs: ' + name)
    print(json.dumps({'status': 'verified', 'payload_files': len(manifest['files']),
                      'base_sha256': base_hash,
                      'scope': 'File integrity only; not a scientific audit or protected-data authorization.'}))


if __name__ == '__main__':
    main()
