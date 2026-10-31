# Data access and splits

No handwriting photographs, labels or downloaded corpora are included. Ask the team for private Drive access. Collection defaults to `/content/drive/MyDrive/CVDL_Handwriting_Dataset/v1`; prototype Cell 4A expects the separately curated `CVDL_Handwriting_Dataset/v1_writers`. Collection page splitting does not itself establish the final writer-held-out assignments; curate/verify those before prototype evaluation.

Cell 4A manifest fields: `sample_id`, `image_path`, `text` (corrected label), `language` (`en`/`zh`), `source_page_id`, `capture_id`, `line_index`, `writer_id`, `writer_name`. Verification adds split and SHA-256 information. Inspect Cell 4A for exact validation and folder conventions. Keep pending captures excluded; derived crops/augmentations must stay in the same split as their writer/source page. Training updates weights; validation selects settings; test is reserved until settings are frozen.

Current code assignments supersede the older project-scope plan:

| Writer | Name | Split |
|---|---|---|
| W01 | Kim | validation |
| W02 | Reiner | test |
| W03 | Yu Wei | test |
| W04 | Wei Kang | train |
| W05 | Wei En | test |

Saved snapshot: 580 English training lines, 35 pages, W04 only; zero curated validation/test lines. Assignments are configured, not evidence that all writers have submitted data.

Public OCR sources: `Teklia/IAM-line` (English) and `Teklia/CASIA-HWDB2-line` (Chinese) on Hugging Face. Obtain access under each dataset's terms. Cells 5-6 inspect official splits and cache 512 training and 64 validation lines per language in private Drive; test is loaded only in Cell 19A. Public dataset revisions were not recorded.

Translation: gated `openlanguagedata/flores_plus` on Hugging Face, CC-BY-SA-4.0 as recorded by Cell 19E. Accept access conditions and use your own read token. Only `dev/eng_Latn.jsonl` and `dev/cmn_Hans.jsonl` at the revision recorded in README are downloaded; the 200 paired references and full benchmark JSON stay private in Drive. Published files contain aggregate scores and settings only.
