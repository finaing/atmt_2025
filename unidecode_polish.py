from unidecode import unidecode

files = [
    "pl-en/data/raw/train.pl",
    "pl-en/data/raw/test.pl",
    "pl-en/data/raw/valid.pl"
]

for file in files:
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines_ascii = [unidecode(line) for line in lines]

    with open(file, "w", encoding="utf-8") as f:
        f.writelines(lines_ascii)

    print(f"✅ Converted {file} to ASCII.")
