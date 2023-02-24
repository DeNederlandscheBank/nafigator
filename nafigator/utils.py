# -*- coding: utf-8 -*-

"""
Utils module.

This module contains utility functions for nafigator package

"""

import io
import logging
from lxml import etree
import re


def load_dtd(dtd_url: str) -> etree.DTD:
    """Utility function to load the dtd

    Args:
        dtd_url: the location of the dtd file

    Returns:
        etree.DTD: the dtd object to be used for validation

    """
    dtd = None
    r = open(dtd_url)
    if r:
        dtd_file_object = io.StringIO(r.read())
        dtd = etree.DTD(dtd_file_object)
    if dtd is None:
        logging.error("failed to load dtd from" + str(dtd_url))
    else:
        logging.info("Succesfully to load dtd from" + str(dtd_url))
    return dtd


def normalize_token_orth(orth: str) -> str:
    """
    Function that normalizes the token text

    Args:
        orth: the token text to be normalized

    Returns:
        str: the normalized token text

    """
    if "\n" in orth:
        return "NEWLINE"
    else:
        return remove_control_characters(orth)


def prepare_comment_text(text: str) -> str:
    """
    Function to prepare comment text for xml

    Args:
        text: comment to be converted to xml comment

    Returns:
        str: converted comment text

    """
    text = text.replace("--", "DOUBLEDASH")
    if text.endswith("-"):
        text = text[:-1] + "SINGLEDASH"
    return text

# TODO: check if function is useful or remove
# def remove_illegal_chars(text):
#     return re.sub(illegal_pattern, "", text)

# Only allow legal strings in XML:
# http://stackoverflow.com/a/25920392/2899924
# illegal_pattern = re.compile(
#     "[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\u10000-\u10FFFF]+"
# )
# improved version:


# A regex matching the "invalid XML character range"
ILLEGAL_XML_CHARS_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1F\uD800-\uDFFF\uFFFE\uFFFF]"
)


def remove_illegal_chars(text: str) -> str:
    """
    Function to remove illegal characters in text

    Args:
        text: string from which illegal characters need to be removed

    Returns:
        str: string with removed illegal characters

    """
    if text is not None:
        return re.sub(ILLEGAL_XML_CHARS_RE, "", text)
    else:
        return None


def remove_control_characters(html: str) -> str:
    """
    Function to strip invalid XML characters that `lxml` cannot parse.

    type: (t.Text) -> t.Text

    See: https://github.com/html5lib/html5lib-python/issues/96

    The XML 1.0 spec defines the valid character range as:
    Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] | [#x10000-#x10FFFF]

    We can instead match the invalid characters by inverting that range into:
    InvalidChar ::= #xb | #xc | #xFFFE | #xFFFF | [#x0-#x8] | [#xe-#x1F] | [#xD800-#xDFFF]

    Sources:
    https://www.w3.org/TR/REC-xml/#charsets,
    https://lsimons.wordpress.com/2011/03/17/stripping-illegal-characters-out-of-xml-in-python/

    Args:
        html: text from which control characters need to be removed

    Returns:
        str: string with removed control characters

    """

    def strip_illegal_xml_characters(s, default, base=10):
        # Compare the "invalid XML character range" numerically
        n = int(s, base)
        if (
            n in (0xB, 0xC, 0xFFFE, 0xFFFF)
            or 0x0 <= n <= 0x8
            or 0xE <= n <= 0x1F
            or 0xD800 <= n <= 0xDFFF
        ):
            return ""
        return default

    # We encode all non-ascii characters to XML char-refs, so for example "💖" becomes: "&#x1F496;"
    # Otherwise we'd remove emojis by mistake on narrow-unicode builds of
    # Python
    html = html.encode("ascii", "xmlcharrefreplace").decode("utf-8")
    html = re.sub(
        r"&#(\d+);?",
        lambda c: strip_illegal_xml_characters(c.group(1), c.group(0)),
        html,
    )
    html = re.sub(
        r"&#[xX]([0-9a-fA-F]+);?",
        lambda c: strip_illegal_xml_characters(c.group(1), c.group(0), base=16),
        html,
    )
    html = ILLEGAL_XML_CHARS_RE.sub("", html)
    return html


# TODO: move with tembasedprocessor and term_extraction, if those modules are removed from package
def sublist_indices(sub, full):
    """
    OUTDATED - CHECK TO REMOVE
    Returns a list of indices of the full list that contain the sub list
    :param sub: list
    :param full: list
    :return: list

    >>> sublist_indices(['Felix'], ['De', 'kat', 'genaamd', 'Felix', 'eet', 'geen', 'Felix'])
    [[3], [6]]
    >>> sublist_indices(
            ['Felix', 'Maximiliaan'],
            ['De', 'kat', 'genaamd', 'Felix', 'Maximiliaan', 'eet', 'geen', 'Felix']
        )
    [[3, 4]]
    """
    if sub == []:
        return []
    if full == []:
        return []
    found = []
    for idx, item in enumerate(full):
        if item == sub[0]:
            if len(sub) == 1:
                found.append([idx])
            else:
                match = True
                for i, s in enumerate(sub[1:]):
                    if len(full) > idx + i + 1:
                        if s != full[idx + i + 1]:
                            match = False
                    else:
                        match = False
                if match:
                    found.append(list(range(idx, idx + len(sub))))
    return found


def remove_sublists(lst):
    """
    Returns a list where all sublists are removed
    :param lst: list
    :return: list

    >>> remove_sublists([[1, 2, 3], [1, 2]])
    [[1, 2, 3]]
    >>> remove_sublists([[1, 2, 3], [1]])
    [[1, 2, 3]]
    >>> remove_sublists([[1, 2, 3], [1, 2], [1]])
    [[1, 2, 3]]
    >>> remove_sublists([[1, 2, 3], [2, 3, 4], [2, 3], [3, 4]])
    [[1, 2, 3], [2, 3, 4]]
    """
    curr_res = []
    result = []
    for elem in sorted(map(set, lst), key=len, reverse=True):
        if not any(elem <= req for req in curr_res):
            curr_res.append(elem)
            r = list(elem)
            result.append(r)
    return result
