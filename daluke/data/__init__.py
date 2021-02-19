import bz2
import json
import os
import shutil
import wget
import xml.etree.ElementTree as ET
from urllib.request import urlopen

from pelutils import log
from tqdm import tqdm

_FNAME_REPLACES = { "/", "\\", ":" }  # These can cause filename problems
_FNAME = "%swiki-latest-pages-articles.xml.bz2"
_DOWNLOAD_URL = "https://dumps.wikimedia.org/%swiki/latest/"
_fname = lambda lang: _FNAME % lang
_dl_url = lambda lang: _DOWNLOAD_URL % lang + _fname(lang)

_TAGS = { "title", "redirect", "text" }

def _fix_fname(fname: str) -> str:
    for rep in _FNAME_REPLACES:
        fname = fname.replace(rep, "_")
    return fname

def _download_wikidata(dest: str, lang="da", force=False):
    """ Downloads wikipedia as .xml.bz2 """
    os.makedirs(dest, exist_ok=True)
    dest = os.path.join(dest, _fname(lang))
    if os.path.exists(dest):
        if force:
            os.remove(dest)
        else:
            log("%s already exists, so not downloading" % dest)
            return
    log("Downloading %s Wikipedia to %s" % (lang, dest))

    url = _dl_url(lang)

    # Get size of download and warn user of size if >=1GB
    site = urlopen(url)
    dlsize = int(site.info()["Content-Length"]) / 2 ** 30  # :*
    if dlsize >= 1:
        cont = log.bool_input(
            log.input("Download of compressed file is %.2f GB. Continue? [y/N] " % dlsize),
            default=False,
        )
        if not cont:
            return

    # Download
    wget.download(url, out=dest)
    print()  # wget.download has no newline character

def _parse_wikidata(dest: str, lang="da"):
    """ Splits .xml.bz2 file downloaded by download_wikipedia into pages. Original file is deleted """
    xmlfile = os.path.join(dest, _fname(lang))
    dest = os.path.join(dest, "pages_%s" % lang)
    if os.path.exists(dest):
        log.warning("Deleting content of %s" % dest)
        shutil.rmtree(dest)
    os.makedirs(dest)

    log("Reading %s" % xmlfile)
    with bz2.BZ2File(xmlfile) as xmlobj:
        page_info: dict = None
        for event, elem in tqdm(ET.iterparse(xmlobj, events=("start", "end",))):
            tag = elem.tag.split("}")[-1]
            if event == "start" and tag == "page":
                page_info = dict()
            elif event == "end" and tag == "page":
                for tag_ in _TAGS:
                    if tag_ not in page_info:
                        page_info[tag_] = None
                with open(os.path.join(dest, _fix_fname(page_info["title"])+".json"), "w") as dump:
                    log.debug("Dumping '%s'" % page_info["title"], with_print=False)
                    json.dump(page_info, dump)
                page_info = None
            # For some reason, text is a rare few times given at "end" instead of "start", so no event check
            elif tag in _TAGS and page_info is not None and elem.text:
                page_info[tag] = elem.text
            elem.clear()
            # TODO: Maybe consider deleting empty references? Probably not a problem (thread is outdated python 2 code)
            # https://stackoverflow.com/questions/29401068/parsing-a-large-bz2-file-40-gb-with-lxml-iterparse-in-python-error-that-does

    os.remove(xmlfile)


def download_wikipedia(dest: str, lang="da", force_download=False):
    """ Download Wikipedia in a given language to dest folder """
    _download_wikidata(dest, lang, force=force_download)
    _parse_wikidata(dest, lang)


if __name__ == "__main__":
    log.configure("local_wikida/wikida.log", "Downloading Danish Wikipedia")
    download_wikipedia("local_wikida", force_download=True)

