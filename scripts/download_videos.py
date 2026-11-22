"""Download the clips listed in videos.csv into data/videos/."""
import csv
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

for row in csv.DictReader((ROOT / "videos.csv").open()):
    target = ROOT / "data" / "videos" / f"{row['clip']}.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        request = urllib.request.Request(row["file"], headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.pexels.com/"})
        target.write_bytes(urllib.request.urlopen(request).read())
    print(target.name, target.stat().st_size)
