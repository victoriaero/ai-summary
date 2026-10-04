from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = PROJECT_ROOT / "resources" / "epistemic_commitment"

RESOURCES = {
    "bioscope": {
        "url": (
            "https://rgai.inf.u-szeged.hu/sites/rgai.sed.hu/files/"
            "bioscope.zip"
        ),
        "sha256": (
            "b20b525506f783ab9c120548ac4022ab587bf7cc419beaf759f7253f0cf0b1d1"
        ),
        "archive": RESOURCE_ROOT / "bioscope" / "bioscope.zip",
        "destination": RESOURCE_ROOT / "bioscope" / "raw",
    },
    "megaveridicality_v2_1": {
        "url": (
            "https://megaattitude.io/projects/mega-veridicality/"
            "mega-veridicality-v2.1.zip"
        ),
        "sha256": (
            "1442ef1f0da66b4965f6d289966bd6240082ce0959c56584b4224ec00eb82219"
        ),
        "archive": (
            RESOURCE_ROOT
            / "megaveridicality_v2_1"
            / "mega-veridicality-v2.1.zip"
        ),
        "destination": RESOURCE_ROOT / "megaveridicality_v2_1" / "raw",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ai-summary-resource-downloader/1.0"},
    )
    with urllib.request.urlopen(request) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out)
    temporary.replace(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(resolved_destination):
                raise ValueError(
                    f"Unsafe archive member in {archive}: {member.filename}"
                )
        zipped.extractall(destination)


def main() -> None:
    for name, specification in RESOURCES.items():
        archive = specification["archive"]
        if not archive.exists():
            print(f"Downloading {name} from {specification['url']}")
            download(str(specification["url"]), archive)
        observed = sha256(archive)
        if observed != specification["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {archive}: {observed}; "
                f"expected {specification['sha256']}"
            )
        safe_extract(archive, specification["destination"])
        print(f"Verified and extracted {name}: {observed}")


if __name__ == "__main__":
    main()
