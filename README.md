from concurrent.futures import ThreadPoolExecutor, as_completed

def download_single(gid):
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
            if r.status_code == 200 and len(r.text) > 1000:
                out_path.write_text(r.text, encoding="utf-8")
                return
        except:
            pass


def download_books(ids, max_workers=20):
    print(f"Downloading {len(ids)} books with {max_workers} threads...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(download_single, gid) for gid in ids]

        for _ in tqdm(as_completed(futures), total=len(futures)):
            pass