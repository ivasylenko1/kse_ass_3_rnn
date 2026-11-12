"""
Apply the labeling-convention rules to an existing submission (no model needed).
    python apply_rules.py submissions/gru_bpe_e1024_f0_qwen_2.csv
writes the same file name with _rules added.
"""
import json, re, sys
import pandas as pd

TEST_CSV = "Data/test.csv"

def hard_processing(text):                         # copied from the competition
    text = re.sub(r"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r'\d', '', text)
    text = re.sub(r'\b\w\b\s?', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

# ---------------- 6. rules for the test's labeling conventions ----------------
FOOTER = re.compile(r"пряма трансляція|хуйовий київ|надіслати новину|запропонувати новину|підписатис[ья]|життя.?оболонь|украина сейчас", re.I)
ORG_BEFORE = re.compile(r"(президент\w*|збройн\w+ сил\w*|верховн\w+ рад\w*|кабмін\w*|уряд\w*|мвс|сбу|національн\w+ банк\w*|"
                        r"генштаб\w*|міністр\w*|міністерств\w*|прокурор\w*|геро[йя]|громадян\w*|конституці\w+|прем.єр\w*)\s+$", re.I)
STREET = re.compile(r"^(вул|вулиц|просп|проспект|бульв|бульвар|пров|провулок|площ|шосе|ул|улиц|пр-т)", re.I)
HOUSE = re.compile(r"^,?\s*\d+[а-яa-z]?(/\d+)?", re.I)

def apply_rules(text, locs):
    kept = []
    for loc in locs:
        places = [m.start() for m in re.finditer(re.escape(loc), text)]
        line = lambda i: text[text.rfind("\n", 0, i) + 1: (text.find("\n", i) + 1 or len(text) + 1) - 1]
        # rule 1: a string that only appears inside channel footer lines is not a location
        if places and all(FOOTER.search(line(i)) for i in places):
            continue
        # rule 2: "України" right after an institution ("Президент України", "ЗСУ ...") is part of a name, not a place
        if hard_processing(loc) in ("україни", "украины") and places and all(ORG_BEFORE.search(text[max(0, i - 40):i]) for i in places):
            continue
        # rule 3: a street keeps its house number ("вул. Солом'янська, 28"), as in the labeled sample
        if STREET.match(loc) and places:
            m = HOUSE.match(text[places[0] + len(loc):])
            if m: loc = loc + m.group()
        if loc not in kept: kept.append(loc)
    return kept

# ---------------- run ----------------
path = sys.argv[1]
sub = pd.read_csv(path)
test = pd.read_csv(TEST_CSV); text = dict(zip(test.text_id, test.text.astype(str)))
before = sub.locations.map(json.loads)
after = [apply_rules(text[i], locs) for i, locs in zip(sub.text_id, before)]
changed = sum(a != b for a, b in zip(after, before))
sub["locations"] = [json.dumps(a, ensure_ascii=False) for a in after]
out = path.replace(".csv", "_rules.csv")
sub.to_csv(out, index=False)
print(f"{changed} of {len(sub)} posts changed; strings {sum(map(len, before))} -> {sum(map(len, after))}; wrote {out}")