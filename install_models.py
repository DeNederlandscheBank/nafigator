import os
import stanza
from stanza.resources.common import HOME_DIR
from typing import List


def install_stanza_models(languages: List[str] = ["nl", "en"]) -> None:
    """Downloads the stanza models for the specified languages list.

    Args:
        languages (List[str]): List like ["nl", "en"]
    """
    # Download stanza resources if not present
    for language in languages:
        stanza.download(lang=language)
        os.remove(os.path.join(HOME_DIR, f"stanza_resources/{language}/default.zip"))


if __name__ == "__main__":
    install_stanza_models()
