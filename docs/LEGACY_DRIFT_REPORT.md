# Legacy Checkout Drift Report and Temporary Waiver

Date: 2026-07-20

This is a bounded, read-only report for the external Music Critic V1 checkout.
Phase 2B.1 did not modify, format, stage, commit, reset, clean, restore, or
otherwise write that checkout. The report implements closure resolution C: a
temporary waiver while the checkout owner decides whether its staged changes
should become the new recorded snapshot or be restored independently.

## Repository identity

- Recorded and current HEAD: `2d8281f31cc9ad9c8fecaf332da0c61e0e949415`
- Recorded and current branch: `sections`
- Recorded snapshot: `docs/legacy_snapshot.json`, captured
  `2026-07-16T18:11:29Z`
- Current status: 29 staged entries; no unstaged or untracked entries
- Snapshot status: 15 deleted, modified, or untracked entries
- Phase 2B.1 repository diff: contains no files from the external checkout

Hashes below are full Git blob object IDs. “Recorded” means the blob at the
pinned legacy HEAD; `—` means that no blob existed at that path. For renames,
the recorded source and current destination have the same blob.

## Renamed paths

| Removed path | Added path | Recorded blob | Current blob |
|---|---|---|---|
| `docs/FIELDS_DECODE.txt` | `docs/music_critic_v1/FIELDS_DECODE.txt` | `d94c94d44dc52318ee276ef3598cfcc4ce405e66` | `d94c94d44dc52318ee276ef3598cfcc4ce405e66` |
| `docs/hooktheory_processed.txt` | `docs/music_critic_v1/hooktheory_processed.txt` | `77802603e207cf80614305033a760685900d13cb` | `77802603e207cf80614305033a760685900d13cb` |
| `docs/hooktheory_selected_field_types_documentation.txt` | `docs/music_critic_v1/hooktheory_selected_field_types_documentation.txt` | `bfc3ebe15e9f38d2dd99150d75eea7122f554603` | `bfc3ebe15e9f38d2dd99150d75eea7122f554603` |
| `docs/original_songs_timeline.txt` | `docs/music_critic_v1/original_songs_timeline.txt` | `342a8e20545c9328af0562f43f5b4ca617927052` | `342a8e20545c9328af0562f43f5b4ca617927052` |

## Added paths

The four rename destinations above are added paths paired with their removed
sources. The remaining staged additions are:

| Path | Recorded blob | Current blob |
|---|---|---|
| `docs/music_critic_v1/c4_observer_chord_prediction.puml` | — | `85871573e8e668b811f07f28572361bb5344f2bb` |
| `docs/music_critic_v1/eda_figures/01_split_distribution.png` | — | `70178d36bfe460b0e53bb9ef4c79b3cee01beac5` |
| `docs/music_critic_v1/eda_figures/02_origin_coverage.png` | — | `fcadc371da045585dbd3f5887b6627dcaf610d47` |
| `docs/music_critic_v1/eda_figures/03_quality_flags.png` | — | `995b8d40e5f688e9ab384cafeb3f819d2c9e3d87` |
| `docs/music_critic_v1/eda_figures/04_duration_and_density.png` | — | `04339e6300f3a4bc170030ce6c3d52436dab1d59` |
| `docs/music_critic_v1/eda_figures/05_music_metadata.png` | — | `c963ad87f423ea486241b601f4176182d6f7adcb` |
| `docs/music_critic_v1/eda_figures/06_region_counts.png` | — | `2feabdda4953063566456324ee379a49be440ed6` |
| `docs/music_critic_v1/eda_figures/07_chord_distributions.png` | — | `ef7b36a8d37a24c2ca23be1027deb520e7aa4270` |
| `docs/music_critic_v1/eda_figures/08_chord_details.png` | — | `c13e21d19d3fe338ad959451cda8b1a41a13af6c` |
| `docs/music_critic_v1/eda_figures/09_melody_distributions.png` | — | `483f585626d256fa5e056e1e2008c0a95715c29d` |
| `docs/music_critic_v1/eda_figures/10_melody_rest_density.png` | — | `d53f3d55d26023a1c950d9027c2734ead9bb540f` |
| `docs/music_critic_v1/eda_figures/summary.json` | — | `d34706e4a25ecf9eee661fa07e963fced38dd5d5` |
| `docs/music_critic_v1/experiment_inventory.csv` | — | `6ecf68827ea47539d6b916e9dcff3b929ce6d9a5` |
| `docs/music_critic_v1/experiment_inventory.md` | — | `2d0cf695e5b2fedc7ec92b36478448196a86efaf` |
| `docs/music_critic_v2/DECISIONS.md` | — | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` |
| `eda_hooktheory_first6.ipynb` | — | `4b0aae608f3dff6e0afa9fdb913609cf7e7e6fa1` |
| `final_observer_critic_runtime.tar.gz` | — | `7373a628217dc572e1c8b6cce0f6eac3f36ac235` |
| `scripts/create_first6_eda_notebook.py` | — | `36f95fb0f0011040f7976bb58666bf244cacbdf7` |
| `scripts/rebuild_observer_targets_with_intermediates.sh` | — | `30749481a60c0c551eae52528dc8e6111842bafd` |
| `scripts/run_teacher_local_theory_corr32_no_mlm.sh` | — | `843c486b1d9de071b325fec06956ca141589dbff` |
| `scripts/run_teacher_local_theory_mlm20_corr70.sh` | — | `3f2fe0aedad5f1f262f74ecb1fbb4621d65fcc0b` |
| `scripts/summarize_experiments.py` | — | `a49e636e5f903da6d8c866a40e30247b990eaf4f` |

## Modified paths

| Path | Recorded blob | Current blob |
|---|---|---|
| `docs/music_critic_v2/DATA_CONTRACT.md` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | `8e770f5c57627ebbce1b2da88996870d92f488cb` |
| `docs/music_critic_v2/STATUS.md` | `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` | `365b387b1248eafc82f2a2c376492160e36ad7b3` |
| `hk_data_pipeline.ipynb` | `b66db5a4cca42bb8cbdeb6e5ce96f3a1ad99dc2b` | `00df107bd277a9f322426b79395f1a2c2dff58c7` |

## Waiver

The legacy snapshot check remains intentionally failing and visible. This
waiver does not classify the staged changes as expected, does not refresh
`docs/legacy_snapshot.json`, and does not authorize changes to the external
checkout. The repository owner must later choose either resolution A (approve
the external changes and refresh the V2 snapshot deliberately) or resolution B
(restore the external checkout manually and rerun the check).
