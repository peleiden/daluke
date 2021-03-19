import os
import bz2
from typing import Generator

import click
from pelutils import log
from tqdm import tqdm

def repeat_entities(article: str) -> str:
    start_indices = list()
    hyperlinks = list()
    skip = 0
    for i in range(len(article)):
        if skip:
            skip -= 1
            continue
        if article[i:].startswith("[["):
            start_indices.append(i+2)
            skip = 1
        elif article[i:].startswith("]]") and start_indices:
            hl = article[start_indices.pop():i]
            if "[[" not in hl:
                hyperlinks.append(hl)
            skip = 1

    # [ (text, hyperlink) ]
    links = list()
    for link in hyperlinks:
        if not link:
            continue
        split = link.split("|")
        if len(split) > 2:
            continue
        elif len(split) == 2:
            try:
                l, t = link.split("|")
                if not t.strip():
                    continue
                links.append((t.lower(), l.lower()))
            except ValueError:
                continue
        else:
            links.append((link.lower(), link.lower()))
    links = sorted(set(links), key=lambda x: x[0], reverse=True)

    i = 0
    s = ""
    while i < len(article):
        if article[i:i+2] == "[[":
            try:
                next_index = article.index("]]", i+2)
                s += article[i:next_index+2]
                i = next_index + 2
            except ValueError:
                s += article[i:]
                break
        elif article[i].isspace():
            s += article[i]
            i += 1
            for text, link in links:
                article_text = article[i:i+len(text)]  # Keep casing
                if article_text.lower() == text:
                    i += len(link)
                    if text == link:
                        s += f"[[{article_text}]]"
                    else:
                        s += f"[[{link}|{article_text}]]"
                    break
        else:
            s += article[i]
            i += 1

    return s

PREPROCESS_FUNCS = {
    "default": lambda x: x,
    "repeat-entities": repeat_entities,
}

def _get_lineblocks(filepath: str) -> Generator:

    with bz2.BZ2File(filepath) as xmlfile:
        current_lines = list()
        while True:
            try:
                line = next(xmlfile)
            except StopIteration:
                break
            decoded = line.decode("utf8")
            current_lines.append(decoded)
            if decoded.strip().startswith("<text"):
                # Yield non-text
                yield False, "".join(current_lines[:-1])
                # Read until end of text is reached
                lines = list()
                while True:
                    lines.append(decoded)
                    if decoded.endswith("</text>\n"):
                        yield True, "".join(lines)
                        break
                    decoded = next(xmlfile).decode("utf8")
                current_lines = list()
        yield False, "".join(current_lines)

def _replace_bytes(tag: str, nbytes: int) -> str:
    bytes_index = tag.index("bytes=\"")
    end_index = tag.index("\"", bytes_index+7)
    return tag[:bytes_index+7] + str(nbytes) + tag[end_index:]

#FIXME: Remove click
@click.command()
@click.argument("wikidownload", type=click.Path(exists=True, dir_okay=False))
@click.option("--func", default="default")
def preprocess(wikidownload: str, func: str):
    log.configure(os.path.join(os.path.split(wikidownload)[0], "preprocessing.log"), "Preprocessing", log_commit=True)
    log(
        "Wikidump path: %s" % wikidownload,
        "Function:      %s" % func,
    )
    func = PREPROCESS_FUNCS[func]

    dump_file = os.path.splitext(wikidownload)[0] + ".preprocessed.bz2"

    log.section("Beginning preprocessing")
    log("Saving to %s" % dump_file)
    with bz2.BZ2File(dump_file, "a") as dump:
        for is_text, text in tqdm(_get_lineblocks(wikidownload), unit=" blocks"):
            if is_text:
                text_start = text.index(">") + 1
                text_end = -len("</text>\n")
                start_tag = text[:text_start].encode()
                end_tag = text[text_end:].encode()
                text = func(text[text_start:text_end]).encode()
                # start_tag = _replace_bytes(start_tag, len(text)).encode()
                text = start_tag + text + end_tag
            else:
                text = text.encode()
            dump.write(text)

if __name__ == "__main__":
    with log.log_errors:
        # pylint: disable=no-value-for-parameter
        preprocess()