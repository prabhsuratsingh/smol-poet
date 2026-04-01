import os
import tarfile
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm

OUT_DIR = Path("gutenberg_poetry")
RDF_ARCHIVE = OUT_DIR / "rdf-files.tar.bz2"
RDF_DIR = OUT_DIR / "cache" / "epub"
RAW_DIR = OUT_DIR / "raw_poetry"
CORPUS_FILE = OUT_DIR / "poetry_corpus_new.txt"

RDF_URL = "https://www.gutenberg.org/cache/epub/feeds/rdf-files.tar.bz2"

OUT_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

def download_rdf():
    if RDF_ARCHIVE.exists():
        print("RDF archive exists")
        return
    print("Downloading RDF catalog...")
    with requests.get(RDF_URL, stream=True) as r:
        r.raise_for_status()
        with open(RDF_ARCHIVE, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)

def extract_rdf():
    if RDF_DIR.exists():
        print("RDF already extracted")
        return
    print("Extracting RDF...")
    with tarfile.open(RDF_ARCHIVE, "r:bz2") as tar:
        tar.extractall(OUT_DIR)

def find_poetry_ids():
    poetry_ids = []

    print("Scanning RDF for poetry...")

    for root, _, files in os.walk(RDF_DIR):
        for f in files:
            if not f.endswith(".rdf"):
                continue

            path = os.path.join(root, f)

            try:
                text = open(path, encoding="utf-8", errors="ignore").read().lower()

                if not ("poetry" in text or "poems" in text or "verse" in text):
                    continue

                if ">en<" not in text:
                    continue

                gid = f.replace("pg", "").replace(".rdf", "")
                poetry_ids.append(gid)

            except:
                pass

    print("Found poetry ids:", len(poetry_ids))
    return poetry_ids


def download_books(ids):
    print("Downloading poetry books...")
    for gid in tqdm(ids):
        out_path = RAW_DIR / f"{gid}.txt"
        if out_path.exists():
            continue

        urls = [
            f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
            f"https://www.gutenberg.org/files/{gid}/{gid}.txt"
        ]

        for url in urls:
            try:
                r = requests.get(url, timeout=20)
                if r.status_code == 200 and len(r.text) > 500:
                    out_path.write_text(r.text, encoding="utf-8")
                    break
            except:
                pass


def strip_gutenberg(text):
    lower = text.lower()

    start = 0
    end = len(text)

    start_markers = [
        "*** start of the project gutenberg ebook",
        "***start of the project gutenberg ebook"
    ]
    end_markers = [
        "*** end of the project gutenberg ebook",
        "***end of the project gutenberg ebook"
    ]

    for m in start_markers:
        i = lower.find(m)
        if i != -1:
            start = i + len(m)
            break

    for m in end_markers:
        i = lower.find(m)
        if i != -1:
            end = i
            break

    return text[start:end]

def remove_frontmatter(text):
    lines = text.splitlines()

    for i in range(min(len(lines), 300)):
        line = lines[i].strip()
        if not line:
            continue

        if len(line) > 20 and not line.isupper():
            return "\n".join(lines[i:])

    return text

def build_corpus():
    print("Building merged corpus...")
    with open(CORPUS_FILE, "w", encoding="utf-8") as out:
        for file in tqdm(list(RAW_DIR.glob("*.txt"))):
            try:
                txt = file.read_text(encoding="utf-8", errors="ignore")
                txt = strip_gutenberg(txt)
                txt = remove_frontmatter(txt)

                txt = "\n".join(
                    l.strip() for l in txt.splitlines() if l.strip()
                )

                if len(txt) < 200:
                    continue

                out.write("<|poem|>\n")
                out.write(txt.strip())
                out.write("\n<|endpoem|>\n\n")
            except:
                pass

    print("Corpus saved:", CORPUS_FILE)


if __name__ == "__main__":
    download_rdf()
    extract_rdf()
    ids = find_poetry_ids()
    download_books(ids)
    build_corpus()

    print("\nDone 🎉")
    print("Dataset:", CORPUS_FILE)