"""
Augmentation round 3: Kyiv content (the gap that adversarial validation found).
Reads long_aug_all.parquet, writes long_aug3_all.parquet with the SAME rows and folds:
  1. substitution: some locations after a preposition are replaced by a Kyiv district, in the right grammatical case
  2. templates: some posts get one extra Kyiv sentence at the start (street + house number, metro station, district)
  3. capitals: in some posts one line is written in CAPITALS, as Telegram headlines often are
Every location offset is recomputed, and the script checks that each label still points at the right text.
Run:  python make_aug3.py
"""
import random, re
import pandas as pd

IN_FILE, OUT_FILE = "data/splits/long_aug_all.parquet", "data/splits/long_aug3_all.parquet"
P_SUB, MAX_SUB = 0.3, 2        # chance to replace an eligible location; at most 2 per post
P_TEMPLATE = 0.08              # chance to add one Kyiv sentence to a post
P_CAPS = 0.05                  # chance to capitalise one line
rnd = random.Random(42)

# ---------------- Kyiv names ----------------
# district: (uk genitive, uk preposition + locative, ru genitive, ru preposition + locative)
DISTRICTS = [
    ("Оболоні", "на Оболоні", "Оболони", "на Оболони"),
    ("Троєщини", "на Троєщині", "Троещины", "на Троещине"),
    ("Позняків", "на Позняках", "Позняков", "на Позняках"),
    ("Осокорків", "на Осокорках", "Осокорков", "на Осокорках"),
    ("Виноградаря", "на Виноградарі", "Виноградаря", "на Виноградаре"),
    ("Теремків", "на Теремках", "Теремков", "на Теремках"),
    ("Борщагівки", "на Борщагівці", "Борщаговки", "на Борщаговке"),
    ("Подолу", "на Подолі", "Подола", "на Подоле"),
    ("Печерська", "на Печерську", "Печерска", "на Печерске"),
    ("Русанівки", "на Русанівці", "Русановки", "на Русановке"),
    ("Лук'янівки", "на Лук'янівці", "Лукьяновки", "на Лукьяновке"),
    ("Святошина", "на Святошині", "Святошина", "на Святошине"),
    ("Куренівки", "на Куренівці", "Куреневки", "на Куреневке"),
    ("Дарниці", "на Дарниці", "Дарницы", "на Дарнице"),
    ("Березняків", "на Березняках", "Березняков", "на Березняках"),
    ("Нивок", "на Нивках", "Нивок", "на Нивках"),
    ("Сирця", "на Сирці", "Сырца", "на Сырце"),
    ("Відрадного", "на Відрадному", "Отрадного", "на Отрадном"),
]
METRO_UK = ["Героїв Дніпра", "Почайна", "Тараса Шевченка", "Контрактова площа", "Поштова площа", "Майдан Незалежності",
            "Хрещатик", "Арсенальна", "Дніпро", "Гідропарк", "Лівобережна", "Дарниця", "Чернігівська", "Лісова",
            "Університет", "Театральна", "Вокзальна", "Політехнічний інститут", "Шулявська", "Берестейська", "Нивки",
            "Святошин", "Житомирська", "Академмістечко", "Лук'янівська", "Дорогожичі", "Сирець", "Золоті ворота",
            "Палац спорту", "Кловська", "Печерська", "Звіринецька", "Видубичі", "Славутич", "Осокорки", "Позняки",
            "Харківська", "Вирлиця", "Бориспільська", "Червоний хутір", "Олімпійська", "Либідська", "Деміївська",
            "Голосіївська", "Васильківська", "Виставковий центр", "Іподром", "Теремки"]
METRO_RU = ["Героев Днепра", "Почайна", "Тараса Шевченко", "Контрактовая площадь", "Почтовая площадь",
            "Майдан Независимости", "Крещатик", "Арсенальная", "Днепр", "Гидропарк", "Левобережная", "Дарница",
            "Черниговская", "Лесная", "Университет", "Театральная", "Вокзальная", "Шулявская", "Берестейская", "Нивки",
            "Святошин", "Житомирская", "Академгородок", "Лукьяновская", "Дорогожичи", "Сырец", "Золотые ворота",
            "Дворец спорта", "Кловская", "Печерская", "Выдубичи", "Славутич", "Осокорки", "Позняки", "Харьковская",
            "Бориспольская", "Олимпийская", "Лыбедская", "Демиевская", "Голосеевская", "Васильковская", "Ипподром", "Теремки"]
STREETS_UK = ["вул. Хрещатик", "вул. Велика Васильківська", "вул. Саксаганського", "вул. Антоновича", "вул. Січових Стрільців",
              "вул. Богдана Хмельницького", "вул. Володимирська", "вул. Ярославів Вал", "вул. Лаврська", "вул. Жилянська",
              "вул. Солом'янська", "вул. Борщагівська", "вул. Драгоманова", "вул. Ревуцького", "вул. Закревського",
              "вул. Здолбунівська", "вул. Кирилівська", "вул. Межигірська", "вул. Мечникова", "вул. Петра Сагайдачного",
              "просп. Степана Бандери", "просп. Науки", "просп. Академіка Глушкова", "просп. Голосіївський",
              "просп. Повітрофлотський", "просп. Оболонський", "просп. Валерія Лобановського", "просп. Броварський",
              "бульв. Тараса Шевченка", "бульв. Лесі Українки", "бульв. Дружби Народів", "бульв. Чоколівський"]
STREETS_RU = ["ул. Крещатик", "ул. Большая Васильковская", "ул. Саксаганского", "ул. Антоновича", "ул. Владимирская",
              "ул. Богдана Хмельницкого", "ул. Жилянская", "ул. Соломенская", "ул. Драгоманова", "ул. Кирилловская",
              "просп. Степана Бандеры", "просп. Науки", "просп. Академика Глушкова", "просп. Голосеевский",
              "просп. Воздухофлотский", "просп. Оболонский", "просп. Броварской", "бульв. Леси Украинки"]

# templates: {addr} street + house number, {metro} station, {dloc} preposition + district, {dgen} district (genitive)
TEMPLATES_UK = ["ДТП на {addr}: рух ускладнено.", "На {addr} через аварію на водогоні немає води.",
                "Біля станції метро «{metro}» сталася бійка.", "Сьогодні вночі {dloc} було чутно вибухи.",
                "Поліція перекрила рух на {addr}.", "Мешканці {dgen} скаржаться на відключення світла.",
                "Вхід до станції метро «{metro}» тимчасово закритий.", "{dloc} відкрили новий пункт незламності.",
                "Затори на {addr} у напрямку {dgen}.", "На станції метро «{metro}» затримали чоловіка."]
TEMPLATES_RU = ["ДТП на {addr}: движение затруднено.", "На {addr} из-за аварии нет воды.",
                "Возле станции метро «{metro}» произошла драка.", "Сегодня ночью {dloc} были слышны взрывы.",
                "Полиция перекрыла движение на {addr}.", "Жители {dgen} жалуются на отключение света.",
                "Вход на станцию метро «{metro}» временно закрыт.", "Пробки на {addr} в сторону {dgen}."]

PREP_LOC = {"uk": re.compile(r"(?<!\w)(у|в|на)\s+$", re.I), "ru": re.compile(r"(?<!\w)(в|во|на)\s+$", re.I)}
PREP_GEN = {"uk": re.compile(r"(?<!\w)(з|із|зі|до|біля|від|поблизу)\s+$", re.I), "ru": re.compile(r"(?<!\w)(из|до|возле|около|от)\s+$", re.I)}

def house():
    return str(rnd.randint(1, 150)) + rnd.choice(["", "", "", "А", "Б", "/2"])

def fill(template, lang):
    """Fill a template; return the sentence and the (start, end) of every location in it."""
    d = rnd.choice(DISTRICTS)
    gen, loc = (d[0], d[1]) if lang == "uk" else (d[2], d[3])
    value = {"addr": f"{rnd.choice(STREETS_UK if lang == 'uk' else STREETS_RU)}, {house()}",
             "metro": rnd.choice(METRO_UK if lang == "uk" else METRO_RU), "dgen": gen, "dloc": loc}
    out, spans = "", []
    for piece in re.split(r"(\{\w+\})", template):
        if piece.startswith("{"):
            key, v = piece[1:-1], value[piece[1:-1]]
            if out == "": v = v[0].upper() + v[1:]                    # sentence start
            skip = len(v.split(" ", 1)[0]) + 1 if key == "dloc" else 0  # the preposition is not part of the location
            spans.append((len(out) + skip, len(out) + len(v))); out += v
        else:
            out += piece
    return out, spans

def substitute(text, locs, others, lang):
    """Replace up to MAX_SUB single-word locations that follow a preposition by a Kyiv district in the same case."""
    done = 0
    for i in sorted(range(len(locs)), key=lambda i: -locs[i][0]):        # right to left: earlier offsets stay valid
        a, b = locs[i]
        if done == MAX_SUB or rnd.random() > P_SUB or " " in text[a:b] or not text[a:b][:1].isupper():
            continue
        if any(x < b and y > a for x, y in others):                        # never touch an ORG/PER span
            continue
        if any(j != i and x < b and y > a - 4 for j, (x, y) in enumerate(locs)):   # nor a duplicate/overlapping label
            continue
        d = rnd.choice(DISTRICTS)
        m_loc, m_gen = PREP_LOC[lang].search(text[:a]), PREP_GEN[lang].search(text[:a])
        if m_loc:                                                          # "у Львові" -> "на Оболоні"
            new = d[1] if lang == "uk" else d[3]
            s, prep = m_loc.start(), new.split(" ", 1)[0]
            if text[s].isupper(): new = new[0].upper() + new[1:]
        elif m_gen:                                                        # "до Львова" -> "до Оболоні"
            new = d[0] if lang == "uk" else d[2]
            s, prep = a, ""
        else:
            continue
        delta = len(new) - (b - s)
        text = text[:s] + new + text[b:]
        start = s + (len(prep) + 1 if prep else 0)
        locs[i] = [start, s + len(new)]
        for lst in (locs, others):
            for j, (x, y) in enumerate(lst):
                if x >= b and not (lst is locs and j == i): lst[j] = [x + delta, y + delta]
        done += 1
    return text, locs, others, done

# ---------------- apply ----------------
df = pd.read_parquet(IN_FILE)
texts, locs_out, orgs_out, pers_out, kinds = [], [], [], [], []
for text, lang, lm, om, pm in zip(df.text, df.lang, df.loc_markers, df.org_markers, df.per_markers):
    locs = [[int(a), int(b)] for a, b in lm]
    org = [[int(a), int(b)] for a, b in om]; per = [[int(a), int(b)] for a, b in pm]
    kind = []
    # 1. substitution
    others = org + per
    text, locs, others, n = substitute(text, locs, others, lang)
    org, per = others[:len(org)], others[len(org):]
    if n: kind.append("sub")
    # 2. template sentence at the start
    if rnd.random() < P_TEMPLATE:
        sent, spans = fill(rnd.choice(TEMPLATES_UK if lang == "uk" else TEMPLATES_RU), lang)
        prefix = sent + rnd.choice([" ", "\n", "\n\n"]); d = len(prefix)
        text = prefix + text
        locs = [list(s) for s in spans] + [[a + d, b + d] for a, b in locs]
        org = [[a + d, b + d] for a, b in org]; per = [[a + d, b + d] for a, b in per]
        kind.append("template")
    # 3. one line in capitals (only when capitalising keeps the length, so offsets stay valid)
    if rnd.random() < P_CAPS:
        lines = text.split("\n"); k = rnd.randrange(len(lines))
        if len(lines[k].upper()) == len(lines[k]):
            lines[k] = lines[k].upper(); text = "\n".join(lines); kind.append("caps")
    texts.append(text); locs_out.append(locs); orgs_out.append(org); pers_out.append(per); kinds.append(",".join(kind))

out = df.copy()
out["text"], out["loc_markers"], out["org_markers"], out["per_markers"], out["aug3"] = texts, locs_out, orgs_out, pers_out, kinds
out["locations"] = [[t[a:b] for a, b in m] for t, m in zip(out.text, out.loc_markers)]
out["n_loc"] = out.loc_markers.map(len)

# checks: unchanged posts keep their labels; no new broken labels (empty, or starting/ending with a space).
# The source data already has a few broken labels (noisy automatic labels), so we compare with it.
broken = lambda col: sum(1 for locs in col for s in locs if not s or s != s.strip())
same = (out.aug3 == "").to_numpy()
assert all(list(a) == list(b) for a, b in zip(out.locations[same], df.locations[same])), "unchanged posts changed"
print(f"broken labels: {broken(df.locations)} in the source, {broken(out.locations)} after augmentation")
assert broken(out.locations) <= broken(df.locations), "the augmentation broke labels"
out.to_parquet(OUT_FILE, index=False)

for k in ("sub", "template", "caps"):
    print(f"{k:9s} {out.aug3.str.contains(k).mean():.1%} of posts")
print("posts with a Kyiv name after:", out.text.str.contains("Київ|Києв|Киев|Оболон|Троє|Троещ|Позняк|Осокор|метро|вул\\.|ул\\.").mean().round(3),
      "before:", df.text.str.contains("Київ|Києв|Киев|Оболон|Троє|Троещ|Позняк|Осокор|метро|вул\\.|ул\\.").mean().round(3))
for t, l in out[out.aug3.str.contains("sub|template")].sample(4, random_state=1)[["text", "locations"]].values:
    print("\n", repr(t[:220]), "\n  ->", list(l)[:6])
print("\nsaved", OUT_FILE)