"""Download only frozen Day1 source/development files; preserve originals."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
import sys
import time
import traceback

import requests

ROOT = Path(__file__).resolve().parents[1]


def main():
    start = time.monotonic()
    result = {"started_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(), "files": [], "status": "running"}
    output = ROOT / "result/day1/download_receipt.json"
    output.write_text(json.dumps(result, indent=2))
    try:
        role_path = ROOT / "research/PARTICIPANT_ROLES_v1.csv"
        expected = role_path.with_suffix('.csv.sha256').read_text().split()[0]
        assert hashlib.sha256(role_path.read_bytes()).hexdigest() == expected
        roles = {int(x['subject_id']): x for x in csv.DictReader(role_path.open())}
        plan = json.loads((ROOT / 'research/day1/DOWNLOAD_PLAN.json').read_text())
        assert len(plan) == 4 and sum(x['bytes'] for x in plan) < 3_000_000_000
        checksum_path = ROOT / 'research/day1/sources/100542.md5'
        if not checksum_path.exists():
            url = 'https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/100542.md5'
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            checksum_path.write_bytes(response.content)
        checksum_lines = checksum_path.read_text().splitlines()
        for item in plan:
            relative = item['key'].split('/100542/', 1)[1]
            sid = int(relative.split('/')[1][1:])
            assert roles[sid]['role'] in ['source', 'development'] and roles[sid]['day1_selected'] == '1'
            matches = [line for line in checksum_lines if relative in line]
            assert len(matches) == 1, (relative, len(matches))
            published_md5 = matches[0].split()[0]
            assert len(published_md5) == 32, matches[0]
            path = ROOT / 'data/raw/lee2019' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            began = time.monotonic()
            record = dict(item, path=str(path.relative_to(ROOT)), published_md5=published_md5, subject_role=roles[sid]['role'])
            if path.exists():
                raise FileExistsError(f'Refusing unverified overwrite: {path}')
            partial = path.with_suffix('.mat.part')
            if partial.exists():
                raise FileExistsError(f'Preserve partial download before retry: {partial}')
            md5, sha256 = hashlib.md5(), hashlib.sha256()
            count = 0
            with requests.get(item['url'], stream=True, timeout=(30, 60)) as response:
                response.raise_for_status()
                assert response.headers.get('ETag') == item['etag']
                with partial.open('xb') as f:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        count += len(chunk)
                        if count > item['bytes']:
                            raise ValueError('Remote bytes exceed frozen inventory')
                        f.write(chunk); md5.update(chunk); sha256.update(chunk)
            assert count == item['bytes'], (count, item['bytes'])
            assert md5.hexdigest() == published_md5, relative
            partial.rename(path)
            record.update(download_seconds=time.monotonic()-began, actual_bytes=count, md5=md5.hexdigest(), sha256=sha256.hexdigest(), validation='published MD5 and inventory size/ETag match')
            result['files'].append(record)
            output.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps({'completed':relative,'bytes':count,'seconds':record['download_seconds']}),flush=True)
        result['status'] = 'completed'
        result['exit_code'] = 0
    except Exception as error:
        result.update(status='error', exit_code=1, error=repr(error))
        traceback.print_exc()
    finally:
        result.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-start)
        output.write_text(json.dumps(result, indent=2)+'\n')
    return result['exit_code']


if __name__ == '__main__':
    sys.exit(main())
