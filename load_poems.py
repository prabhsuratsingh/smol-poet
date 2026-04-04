import os
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

lock = threading.Lock()

OUT_DIR = Path("E:/gutenberg_books")
RDF_ARCHIVE = OUT_DIR / "rdf-files.tar.bz2"
RDF_DIR = OUT_DIR / "cache" / "epub"
RAW_DIR = OUT_DIR / "raw_books"
CORPUS_FILE = OUT_DIR / "books_corpus.txt"

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

def find_english_ids():
    ids = []

    print("Scanning RDF for English books...")

    for root, _, files in os.walk(RDF_DIR):
        for f in files:
            if not f.endswith(".rdf"):
                continue

            path = os.path.join(root, f)

            try:
                text = open(path, encoding="utf-8", errors="ignore").read().lower()

                # Keep ONLY English
                if ">en<" not in text:
                    continue

                # Optional: skip very small / low-quality entries
                if "sound" in text or "audiobook" in text:
                    continue

                if "illustrated" in text:
                    continue

                if "collection" in text and "poems" not in text:
                    continue

                gid = f.replace("pg", "").replace(".rdf", "")
                ids.append(gid)

            except:
                pass

    print("Found English book ids:", len(ids))
    return ids

# def download_books(ids):
#     print("Downloading poetry books...")
#     for gid in tqdm(ids):
#         out_path = RAW_DIR / f"{gid}.txt"
#         if out_path.exists():
#             continue

#         urls = [
#             f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
#             f"https://www.gutenberg.org/files/{gid}/{gid}.txt"
#         ]

#         for url in urls:
#             try:
#                 r = requests.get(url, timeout=20)
#                 if r.status_code == 200 and len(r.text) > 500:
#                     out_path.write_text(r.text, encoding="utf-8")
#                     break
#             except:
#                 pass

def download_one(gid):
    out_path = RAW_DIR / f"{gid}.txt"
    if out_path.exists():
        return

    urls = [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt"
    ]

    for url in urls:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200 and len(r.text) > 500:
                with lock: 
                    out_path.write_text(r.text, encoding="utf-8")
                return
        except:
            pass


def download_books(ids, max_workers=20):
    print(f"Downloading with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_one, gid) for gid in ids]

        for _ in tqdm(as_completed(futures), total=len(futures)):
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

                out.write("<|book|>\n")
                out.write(txt.strip())
                out.write("\n<|endbook|>\n\n")
            except:
                pass

    print("Corpus saved:", CORPUS_FILE)


if __name__ == "__main__":
    download_rdf()
    extract_rdf()
    ids = find_english_ids()
    download_books(ids, max_workers=30)
    build_corpus()

    print("\nDone 🎉")
    print("Dataset:", CORPUS_FILE)