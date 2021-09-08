from __future__ import annotations
import bz2
import multiprocessing as mp
import os
import re
import shutil
import tempfile
from typing import Generator

import click
from pelutils import log
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from daluke.pretrain.data import load_entity_vocab, ignore_title


MIN_ENT_LEN = 3
MAX_ENT_LEN = 30
_file_replacements = {
    ":": "__COLON__",
    "/": "__SLASH__",
}
def fix_filename(fname):
    for fro, to in _file_replacements.items():
        fname = fname.replace(fro, to)
    return fname

def default(_):
    pass

def repeat_entities(args):
    fname: str = args[0]
    entity_vocab: set[str] = args[1]

    with open(fname) as f:
        article = " ".join(f.read().split())
        article = article.replace("'''", "")

    space_indices = (i for i, c in enumerate(article) if c == " ")

    i = 0
    s = ""
    while i < len(article):
        # Check if already annotated - this happens in the wikipedia dataset
        if article[i:i+2] == "[[" and fname.endswith("wiki"):
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
            for j in range(MAX_ENT_LEN, MIN_ENT_LEN-1, -1):
                if article[i:i+j].lower() in entity_vocab:
                    s += f"[[{article[i:i+j]}]]"
                    i += j
            else:
                s += article[i:i+MAX_ENT_LEN]

        # Jump to next space
        next_space_index = next(space_indices, -1)
        while i < next_space_index:
            next_space_index = next(space_indices, -1)
        if i == -1:
            break

    with open(fname, "w") as f:
        f.write(s)

PREPROCESS_FUNCS = {
    "default": default,
    "repeat-entities": repeat_entities,
}

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
        yield from (os.path.join(root, f) for f in files if re.fullmatch(r"[a-zA-Z]+_[0-9]+", f))

@click.command()
@click.argument("dump-db-file", type=click.Path(exists=True, dir_okay=False))
@click.option("--func", default="default")
@click.option("--entity-vocab-file", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--dagw-sections", type=click.Path(exists=True, file_okay=False), default=None)
def preprocess(dump_db_file: str, func: str, entity_vocab_file: str | None, dagw_sections: str | None):
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
        "Function:      %s" % func,
    )

    log("Loading entity vocab")
    entity_vocab = { e.lower() for e in load_entity_vocab(entity_vocab_file) }

    dagw_files = list()
    if dagw_sections:
        log("Finding gigaword data files")
        dagw_files = list(_get_dagw_files(dagw_sections))
        log("Found %i dagw files" % len(dagw_files))

    tmpdir = "local_tmpdir"
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

    files = [os.path.join(tmpdir, x) for x in os.listdir(tmpdir)]
    log("Saved a total of %i articles to %s" % (len(files), tmpdir))

    log.section("Beginning preprocessing")
    func = PREPROCESS_FUNCS[func]
    log("Using function '%s'" % func.__name__)
    process_map(func, [(f, entity_vocab) for f in files], max_workers=os.cpu_count(), chunksize=1024)

    dump_file = os.path.splitext(dump_db_file)[0] + ".%s.bz2" % func
    log.section("Saving preprocessed files to %s" % dump_file)
    with bz2.BZ2File(dump_file, "w") as dump:
        with bz2.BZ2File(dump_db_file) as old_dump:
            line = b""
            while not line.strip().startswith("<page>"):
                dump.write(line)
                line = old_dump.readline()
        for i, fname in tqdm(enumerate(files)):
            with open(fname) as f:
                text = f.read().encode()
            dump.write("""
                <page>
                    <title>{title}</title>
                    <id>{id}</id>
                    <revision>
                        <text bytes="{bytes}" xml:space="preserve">{text}</text>
                    </revision>
                </page>""".format(
                    title = fname[:-4] if fname.endswith(".wiki") else fname,
                    id    = i+1,
                    bytes = len(text),
                    text  = text,
                )
            )
        dump.write(b"\n</mediawiki>")

    # shutil.rmtree(tmpdir)

if __name__ == "__main__":
    with log.log_errors:
        # pylint: disable=no-value-for-parameter
        preprocess()
