# Supervisor and Wattle handover

Use the public GitHub repository as the single entry point:
<https://github.com/Hanning99/COMP8800_Research_Project>.
Its README links the separate public checkpoint dataset and gives the scene, download, restore and evaluation guide. For Wattle, submit a fixed `/tree/<commit>` link so the reviewed artefact version is unambiguous.

The submission arrangement follows the supplied ANU Individual Computing Projects marking-guide text: deliver the artefact directly to the supervisor and submit an access link on Wattle. Wattle submission and messages to the supervisor are performed by the author; this package does not send either automatically.

The initial release upload was verified and `PUBLISH_RECEIPT.json` records:

- the full GitHub commit SHA and `/tree/<commit>` link;
- the immutable Hugging Face dataset revision and its fixed link;
- the GitHub link to submit to Wattle.

Both repositories are public. The Hugging Face revision pinned in `configs/release.json` identifies the verified benchmark archive; later dataset-card edits do not change the archive payloads in that revision.

The handover identifies the report by its SHA-256, explains the installation environment, provides checkpoint evaluation and training-continuation commands separately, and includes actual verification evidence. It does not claim that all source pretraining was repeated.
