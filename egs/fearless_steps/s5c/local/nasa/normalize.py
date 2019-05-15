import logging
import num2words
import random
import re


number_words = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def convert_num2words(m):
    assert re.match(r'^(\d[,\d.]+|[.][\d,]+|\d)[a-z]*$', m.group(0), flags=re.IGNORECASE)
    m1 = re.match(r'19(\d\d)\.?', m.group(0))
    if m1:
        # Probably a year
        return 'nineteen ' + num2words.num2words(m1.group(1))

    t = re.sub(r'(,|\.$)', '', m.group(1))  # remove commas, and word-end-periods

    if re.search(r'\..+\.', t):
        logging.warning("Multiple periods found in '{}'. We split on the periods instead".format(m.group(1)))

        output = []
        for part in t.split('.'):
            output.append(convert_num2words(re.match(r'\b(\d[,\d.]+|[.][\d,]+|\d)([a-z]*)\b', part)))
        return ' ' + ' point '.join(output) + ' ' + m.group(2) + ' '

    is_ordinal = False
    if re.match(r'st|nd|rd|th', m.group(2), flags=re.IGNORECASE):
        is_ordinal = True

    suffix = ''
    if not is_ordinal and len(m.group(2)):
        suffix = ' ' + m.group(2)

    try:
        s = num2words.num2words(t, to='ordinal' if is_ordinal else 'cardinal')
    except Exception:
        logging.error("Could not convert to words: {}".format(m.group(0)))
        raise

    s = re.sub(r'-', ' ', s)

    if random.random() < 0.1:
        # randomly choose decimal instead of point
        s = re.sub('point', 'decimal', s)

    if random.random() < 0.9:
        # it's not common to say zero point. usually just point is used
        s = re.sub('zero point', 'point', s)

    if random.random() < 0.8:
        # it's more common to say point o than point zero
        s = re.sub('point zero', 'point o', s)

    use_o = False
    if random.random() < 0.3:
        use_o = True

    if re.match(r'^\d+$', m.group(0)) and random.random() < 0.3:
        # some times just read the numbers
        s_str = []
        for d in m.group(0):
            d = int(d)
            if d == 0 and use_o:
                s_str.append("o")
                continue
            s_str.append(number_words[d])
        s = ' '.join(s_str)

    s = s + suffix
    return ' ' + s + ' '


def possibly_add_dash(m):
    assert re.match(r'[a-zA-Z]+', m.group(1))
    assert re.match('\d+', m.group(2))
    assert re.match('[a-zA-Z]+-\d+', m.group(0))
    return m.group(1) + (' ' if random.random() < 0.5 else ' dash ') + m.group(2)



def normalize_text(text):
    text = re.sub(r'\r', '', text)  # remove carriage return
    text = re.sub(r'\s+', ' ', text)  # normalize spaces

    text = re.sub(r'\[(pause|long pause)[.]?\]', ' ', text, flags=re.IGNORECASE)     # remove pauses etc.
    text = re.sub(r'\(laughter\)', ' [laughter] ', text, flags=re.IGNORECASE)  # Normalize laughter mark
    text = re.sub(r'\[garble\]', ' <unk> ', text, flags=re.IGNORECASE)  # Normalize garble
    text = re.sub(r'(\{)?garble\}', ' <unk> ', text, flags=re.IGNORECASE)  # Normalize garble
    text = re.sub(r'\{garble(\})?', ' <unk> ', text, flags=re.IGNORECASE)  # Normalize garble

    text = re.sub(r'\[[^]]+km(ph)?\]', ' ', text, flags=re.IGNORECASE)  # Remove miles to km conversion which are inside square brackets
    text = re.sub(r'\[[^]]+(metre|meter)[s]?\]', ' ', text, flags=re.IGNORECASE)  # Remove miles to meters conversion
    text = re.sub(r'\[[^]]+(m/s|mps)\]', ' ', text, flags=re.IGNORECASE)  # Remove miles to meters conversion

    text = re.sub(r'[$]([\d.,]+)', r'\1 dollars ', text, flags=re.IGNORECASE)  # normalize prices
    text = re.sub(r'([\d.,]+)%', r'\1 percent ', text, flags=re.IGNORECASE)  # normalize percentage

    text = re.sub(r'(\.\d+),', r'\1 ', text)  # Here a comma is used as a punctuation and not as a number separator
    text = re.sub(r',(\s|$)', ' ', text) # remove comma at the end of the word

    text = re.sub(r'--', ' ', text)  # remove multiple dashes

    text = re.sub(r'([\d.]+)-([a-z]+)', r'\1 \2', text, flags=re.IGNORECASE)  # split words with numbers and another word: e.g. 12-deg to 12 deg etc.
    text = re.sub(r'([a-z]+)-([\d.+])', possibly_add_dash, text, flags=re.IGNORECASE)  # split words with another word and number e.g. B-1 to B 1 or B dash 1 etc.
    text = re.sub(r'\b([a-z]+)([\d.,]+)\b', r' \1 \2 ', text, flags=re.IGNORECASE)  # split words with letter and numbers e.g. P22
    text = re.sub(r'\b[+](\d+)', r' plus \1', text, flags=re.IGNORECASE)  # convert '+' symbol to "plus" word: e.g. +0.12 to "plus 0.12" etc.
    text = re.sub(r'\b[-](\d+)', r' minus \1', text, flags=re.IGNORECASE)  # convert '-' symbol to "minus" word: e.g. -0.12 to "minus 0.12" etc.

    text = re.sub(r'[?![\];:"/(){}*]', " ", text, flags=re.IGNORECASE)  # remove punctuations
    text = re.sub(r'\.([a-z]+)', r' \1', text, flags=re.IGNORECASE)  # remove periods at the beginning of words
    text = re.sub(r'[.]+', ' ', text, flags=re.IGNORECASE)  # remove multiple periods

    text = re.sub(r'[_]', "", text, flags=re.IGNORECASE)  # remove underscores

    text = re.sub(r'\b(\d[,\d.]+|[.][\d,]+|\d)([a-z]*)\b', convert_num2words, text, flags=re.IGNORECASE)  # convert numbers to words
    text = re.sub(r'[,]', " ", text, flags=re.IGNORECASE)  # remove commas that were not removed
    text = re.sub(r'\s+', ' ', text, flags=re.IGNORECASE)  # remove multiple spaces
    text = re.sub(r'^ ', '', text, flags=re.IGNORECASE)  # remove spaces at the beginning

    words = []
    for w in text.strip().split():
        if re.search(r'\d', w):
            logging.warning("Word still has digits: {}".format(w))

        w = re.sub(r'^[.]', r'', w, flags=re.IGNORECASE)  # remove periods at the beginning of words
        w = re.sub(r"([^s])[']$", r'\1', w, flags=re.IGNORECASE)  # remove single quotes at the end of words if it is not an 's' (plural possesive)
        text = re.sub(r"^[']", '', w, flags=re.IGNORECASE)  # remove single quotes at the beginning of words
        parts = w.split(".")
        if len(parts) > 2:
            # This is an abbreviation
            pass
        else:
            if len(parts) == 2 and parts[1] == "":
                w = parts[0]  # remove the period

        if w == w.upper():
            # This is a possible acronym
            pass
        else:
            w = w.lower()

        w = re.sub(r'-$|^-', '', w)  # remove dash at the ends of words

        if not re.search('[0-9a-z]', w, flags=re.IGNORECASE):
            # word does not contains numbers or letters and is just pure
            # punctuation mark etc.
            continue
        words.append(w)

    text = " ".join(words)
    text = re.sub(r'[.]-', '. ', text)
    text = re.sub(r'-[.]', ' .', text)

    return text.strip()
