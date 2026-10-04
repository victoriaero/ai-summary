# External resources for epistemic commitment

The analysis uses only externally annotated resources. Run:

```bash
python scripts/download_epistemic_resources.py
```

The downloader obtains and verifies:

- BioScope 1.0 from the HUN-REN–SZTE Research Group on Artificial
  Intelligence. The distribution is CC BY 2.0. SHA-256:
  `b20b525506f783ab9c120548ac4022ab587bf7cc419beaf759f7253f0cf0b1d1`.
- MegaVeridicality v2.1 from the MegaAttitude project. The package includes a
  CC BY-SA 4.0 license. SHA-256:
  `1442ef1f0da66b4965f6d289966bd6240082ce0959c56584b4224ec00eb82219`.

The analysis regenerates `hedge_lexicon.csv` from the BioScope XML files. The
CSV is not edited or supplemented manually. MegaVeridicality scores come from
the official `mega-veridicality-v2.1-normalized.tsv` file.

