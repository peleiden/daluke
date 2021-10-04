from __future__ import annotations
import bz2
import os
import re as reee
import shutil
from typing import Generator

import click
from pelutils import log
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from daluke.pretrain.data import load_entity_vocab, ignore_title

_xml_special_characters = {
    "\"": "quot",
    "&": "amp",
    "'": "apos",
    "<": "lt",
    ">": "gt"
}
_special_character_regex = reee.compile(r"&[a-zA-Z0-9#]+\s{0,1};")
_whitespace_regex = reee.compile(r"\s+")
_illegal_characters_regex = reee.compile(u"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]")

_file_replacements = {
    ":": "__COLON__",
    "/": "__SLASH__",
}
def fix_filename(fname):
    for fro, to in _file_replacements.items():
        fname = fname.replace(fro, to)
    return fname

def _insert_xml_special_characters(s: str) -> str:
    for char, name in _xml_special_characters.items():
        s = s.replace(char, f"&{name};")
    return s

def replace_special_characters_and_whitespace(fname: str):
    """ Replaces HTML special characters and any consecutive whitespace with spaces
    The special wikipedia markup for titles with ''' is also removed
    Finally, XML special characters are replaced with the ambersand syntax """
    with open(fname) as f:
        article = f.read()

    article = reee.sub(_whitespace_regex, " ", article)
    article = reee.sub(_illegal_characters_regex, "", article)

    if fname.endswith(".wiki"):
        # Wikipedia is already in XML format, so only change markup, so title can be recognized as entity
        article = article.replace("'''", "")
    else:
        article = reee.sub(_special_character_regex, " ", article)
        article = _insert_xml_special_characters(article)

    article = reee.sub(_whitespace_regex, " ", article)
    with open(fname, "w") as f:
        f.write(article)

def default(_):
    pass

def repeat_entities(args):
    fname: str = args[0]
    entity_vocab: set[str] = args[1]
    min_ent_len, max_ent_len = args[2], args[3]

    with open(fname) as f:
        article = f.read()

    space_indices = (i for i, c in enumerate(article) if c == " ")

    i = 0
    s = ""
    while i < len(article):
        # Check if already annotated - this happens in the wikipedia dataset
        if article[i:i+2] == "[[" and fname.endswith(".wiki"):
            try:
                # Skip the rest of the annotation
                end_index = article.index("]]", i+2)
                s += article[i:end_index+2]
                i = end_index + 2
            except ValueError:
                # No end of annotation found. In this case, the rest of the article is not further annotated
                s += article[i:]
                break
        else:
            # Check if any entity match. Casing is ignored
            for j in range(max_ent_len, min_ent_len-1, -1):
                if article[i:i+j].lower() in entity_vocab:
                    s += f"[[{article[i:i+j]}]]"
                    i += j
                    break

        # Jump to next space
        start_i = i
        next_space_index = next(space_indices, -1)
        while i > next_space_index and next_space_index != -1:
            next_space_index = next(space_indices, -1)
        i = next_space_index + 1
        if i == 0:
            s += article[start_i:]
            break
        else:
            s += article[start_i:i]

    with open(fname, "w") as f:
        f.write(s)

PREPROCESS_FUNCS = {
    "default": default,
    "repeat-entities": repeat_entities,
}

def func(args):
    replace_special_characters_and_whitespace(args[1])
    PREPROCESS_FUNCS[args[0]](args[1:])

def _get_lineblocks(filepath: str) -> Generator[tuple[bool, str, str | None], None, None]:

    with bz2.BZ2File(filepath) as xmlfile:
        current_lines = list()
        title = None
        while True:
            try:
                line = next(xmlfile)
            except StopIteration:
                break
            decoded = line.decode("utf8")
            current_lines.append(decoded)
            if decoded.strip().startswith("<title>") and decoded.strip().endswith("</title>"):
                title = decoded.strip()[7:-8]
            if decoded.strip().startswith("<text"):
                # Yield non-text
                yield False, "".join(current_lines[:-1]), title
                # Read until end of text is reached
                lines = list()
                while True:
                    lines.append(decoded)
                    if decoded.endswith("</text>\n"):
                        yield True, "".join(lines), title
                        title = None
                        break
                    decoded = next(xmlfile).decode("utf8")
                current_lines = list()
        yield False, "".join(current_lines), None

def _get_dagw_files(path: str, ignore_sections: set[str]={"wiki"}) -> Generator[str, None, None]:
    for root, __, files in os.walk(path):
        if os.path.split(root)[-1] in ignore_sections:
            continue
        yield from (os.path.join(root, f) for f in files if reee.fullmatch(r"%s_.+" % os.path.split(root)[-1], f))

@click.command()
@click.argument("dump-db-file", type=click.Path(exists=True, dir_okay=False))
@click.option("--function", default="default")
@click.option("--entity-vocab-file", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--dagw-sections", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--min-entity-length", type=int, default=5)
@click.option("--max-entity-length", type=int, default=48)
@click.option("--max-articles", type=int, default=None)
def preprocess(
    dump_db_file: str,
    function: str,
    entity_vocab_file: str | None,
    dagw_sections: str | None,
    min_entity_length: int,
    max_entity_length: int,
    max_articles: int | None,
):
    if not entity_vocab_file:
        raise RuntimeError("entity-vocab-file must be given")

    log.configure(
        os.path.join(os.path.split(dump_db_file)[0], "preprocessing.log"),
        "Preprocessing",
        log_commit=True,
    )

    log.section("Collecting data")
    log(
        "Wikidump path: %s" % dump_db_file,
        "Function:      %s" % function,
    )

    log("Loading entity vocab")
    entity_vocab = { _insert_xml_special_characters(e.lower()) for e in load_entity_vocab(entity_vocab_file) }

    dagw_files = list()
    if dagw_sections:
        n_words = 0
        log("Finding gigaword data files")
        dagw_files = list(_get_dagw_files(dagw_sections))
        for dagw_file in tqdm(dagw_files):
            with open(dagw_file) as f:
                n_words += len(f.read().split())
        log("Found %i dagw files containing %i words" % (len(dagw_files), n_words))

    # tempdir is not used, as the temporary files can take up more space than what temporary
    # directories usually allow
    tmpdir = os.path.join(os.path.split(dump_db_file)[0], "tmpdir")
    os.makedirs(tmpdir, exist_ok=True)
    log("Saving all articles to temporary directory %s" % tmpdir)
    for dagw_file in tqdm(dagw_files):
        shutil.copy2(dagw_file, os.path.join(tmpdir, fix_filename(os.path.split(dagw_file)[-1])))
    log("Saving Wikipedia files to temporary directory")
    for is_text, text, title in tqdm(_get_lineblocks(dump_db_file), unit=" blocks"):
        if is_text and not ignore_title(title):
            text_start = text.index(">") + 1
            text_end = -len("</text>\n")
            with open(os.path.join(tmpdir, fix_filename(title)[:100]+".wiki"), "w") as f:
                f.write(text[text_start:text_end])

    files = [os.path.join(tmpdir, x) for x in os.listdir(tmpdir)[:max_articles]]
    log("Saved a total of %i articles to %s" % (len(files), tmpdir))

    log.section("Beginning preprocessing on %i threads" % os.cpu_count())
    process_map(
        func,
        [(function, f, entity_vocab, min_entity_length, max_entity_length) for f in files],
        max_workers=os.cpu_count(),
        chunksize=1024,
    )

    dump_file = os.path.splitext(dump_db_file)[0] + ".%s.bz2" % function
    log.info("Saving preprocessed files to %s" % dump_file)
    with bz2.BZ2File(dump_file, "w") as dump:
        with bz2.BZ2File(dump_db_file) as old_dump:
            line = b""
            while not line.strip().startswith(b"<page>"):
                dump.write(line)
                line = old_dump.readline()
        for i, fname in tqdm(enumerate(files), total=len(files)):
            with open(fname) as f:
                text = f.read()
            s = """
            <page>
                <title>{title}</title>
                <id>{id}</id>
                <revision>
                    <text bytes="{bytes}" xml:space="preserve">{text}</text>
                </revision>
            </page>""".format(
                title = fname,
                id    = i+1,
                bytes = len(text),
                text  = text,
            )
            if i == 0:
                s = s[1:]
            dump.write(s.encode("utf-8"))
        dump.write(b"\n</mediawiki>")

    log.info("Removing temporary files")
    shutil.rmtree(tmpdir)
    log.info("Done preprocessing data")

if __name__ == "__main__":
    with log.log_errors:
        # pylint: disable=no-value-for-parameter
        preprocess()
