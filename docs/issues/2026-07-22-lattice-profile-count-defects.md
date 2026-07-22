---
type: research
status: current
created: 2026-07-22
updated: 2026-07-22
tags: [issue, lattice-profiles, anonymity-counts, count-provenance, rl-ranker, data-quality]
companion: [../specs/RL/interactive-ranker-v2.md,
            ../specs/qa-builder-v2.md]
---

# Issue: deferred lattice-profile count and hierarchy defects

The 2026-07-22 Ranker-v2 count audit found profile-quality defects that should be repaired, but
they no longer block RL delivery. Reward-facing counts are now sourced from each decision's
matched profile row; missing or inadmissible row-local evidence is represented explicitly as
provisional null data. No repair in this issue is approved for automatic application.

The complete human-review record is
[`results/ranker_v2/count_repair/proposed-edits.md`](../../results/ranker_v2/count_repair/proposed-edits.md).
Machine-readable diagnoses and execution evidence are in
[`profile-diagnoses.json`](../../results/ranker_v2/count_repair/profile-diagnoses.json),
[`repair-report.json`](../../results/ranker_v2/count_repair/repair-report.json), and
[`unresolved-queue.jsonl`](../../results/ranker_v2/count_repair/unresolved-queue.jsonl).

## Diagnosed defects

The review covered 52 profiles. The diagnosis classified them as:

- **31 merge-key mismatches:** the old frozen environment looked up a level string across a
  runtime type and used the maximum count, rather than using the count from the matched profile
  row. These are primarily environment count-sourcing defects; some underlying rows also merge
  distinct identities.
- **18 wrong authored-order defects:** the ladder order or level selection needs source-level
  repair rather than count sorting, clipping, or another reward-time workaround.
- **3 wrong count-evidence defects:** level values/order can be retained, but the evidence and
  resulting count need correction.

The proposed edits classify 29 profiles as count-only and 23 as order/fill-changing. The latter
invalidate affected decisions' existing QA support if eventually applied.

## Profiles requiring external evidence

Thirteen profiles cannot be repaired from the available local evidence and must remain
unresolved: `LOC:albania`, `LOC:armenia`, `LOC:central african republic`, `LOC:florida`,
`LOC:georgia`, `LOC:madagascar`, `LOC:namibia`, `LOC:portugal`, `LOC:vermont`,
`drug:acetaminophen`, `drug:ibuprofen`, `medical-procedure:blood tests`, and
`medical-procedure:hemoglobin a1c`.

Do not guess their missing levels, order, counts, or evidence. The exact failure and proposed
next evidence step for each profile is recorded in `proposed-edits.md`.

## Locally evidenced proposals awaiting confirmation

Three proposed edits have complete local DOID descendant evidence:

- `health-condition:adenoma`: replace the non-certifying cellular-proliferation count with the
  DOID-backed count while retaining the level sequence.
- `health-condition:bowel dysfunction`: replace the current ladder with the locally supported
  `intestinal disease` → `gastrointestinal system disease` → `disease of anatomical entity`
  chain and its DOID descendant counts.
- `health-condition:hypertension`: use the locally supported `heart disease` →
  `thoracic disease` → `disease of anatomical entity` chain and its DOID descendant counts.

These are **awaiting confirmation; do not auto-apply**. Their exact counts, grounding records,
and downstream classifications are in `proposed-edits.md` and the offline artifacts under
[`offline-run/`](../../results/ranker_v2/count_repair/offline-run/).

## Unapproved model queue

The unresolved producer queue contains **20 items**, each requiring one proposed request to
`Qwen3.6-35B-A3B`. No request was made and that model use is not approved. The queue includes
surface variants and repair evidence that collapse to fewer unique profile-level problems, so
its 20 items are distinct from the 13 unresolved profiles above. Preserve the queue as evidence;
do not call the model without explicit approval.

## Repair CLI defects

Two code defects were exposed by the offline-only run:

1. `scripts/run_lattice_producer.py` rejects an `--out` path outside
   `data/lattice_profiles/proposed/`, while the task's documented deterministic repair path is
   under `results/ranker_v2/count_repair/`. The guard conflicts with the supported plan path.
2. In the producer graph, `force_model_proposal` is evaluated before `offline_only`; therefore
   offline-only does not dominate forced model routing. Offline execution must fail closed or
   queue the item before any model branch can be selected.

Both deserve focused code fixes and regression tests. They were diagnosed only; neither was
changed as part of the frozen-environment migration.

## Resolution criteria

Resolve this issue only after every proposed profile edit is individually confirmed, all
external-evidence work is explicitly approved and recorded, the CLI defects have regression
tests, and the canonical profile plus embedding index are promoted through the normal validated
artifact workflow. Until then, RL consumes row-local grounded counts and tags missing evidence
as provisional rather than blocking delivery.

## Count-reward provisional gate inventory

<!-- count-reward-provisional-gaps:c25fbc37642562b928957aa8bb1c4e3c7fe205761a0a46509c1f4dcdd93e99f6 -->

The provisional count gate confirmed **68** unresolved level actions for
environment `sha256:4cc754a7143252613d2ef0160d7778580621fd973a32e5a0388da510170ddc8a`. These are existing profile-data
defects, not runtime fallbacks: their entire decisions receive flat-count semantics
until source evidence is repaired and confirmed.

Gap actions by runtime type: `LOC` 12, `drug` 1, `health-condition` 49, `medical-procedure` 6.

Concrete gate inventory:

- `health-condition` / `health-condition:arthritis` / `sha256:650f1c639ace518d1ffc015694e2db60df669884fefe15dc0d34fab4f5be214c` / `sha256:a7abd3240f6759934458e1340a17ee67ebdb63ee2ba88c7ea9dba2d5372f197d` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:migraine` / `sha256:d9454620c1566babc3b8095774ff26fdc2bb61d5f0735ef8e9ea6ee3e35b5f5a` / `sha256:914eb6260d47bbb25e67ec29a5b00d566daced1c2a4be0b35210f6eda3d330c8` / `central nervous system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:arthritis` / `sha256:aeae2708d860f2786ac17773d035f56b45305585cbe440b0a7ff5fa5c05dab5f` / `sha256:57e113c569bb602239ef27a73707bac8caabca936f13d626ff608e1613873060` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:arthritis` / `sha256:aef24ea6ce33f667efea615fbbd91f445336c9404a5279d3de534e40c970fc1e` / `sha256:62351c39c2f183d6bf7feeb225b19de1e92e3412b5a52aa9eb4a1aafda98d51e` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `LOC` / `LOC:vermont` / `sha256:ab7b60c69de16f7e70ae46453372b23241eb943f8a6df45bf7ac7b8337774eab` / `sha256:73422cf5e54527d285a9fc728d7c71b0ed7672efe1247b610bfe9fa2de3a3ba3` / `a city in Australia`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `medical-procedure` / `medical-procedure:knee surgery` / `sha256:f858bc1d147108153a85001253e5b5e56b55a1c080fab2133a053aeacbd915f4` / `sha256:0b0510a5e7ca27aa5dd0d1e56e76b1407ec03238baac79ad9e373313446e2f31` / `orthopedic surgical procedures`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `medical-procedure` / `medical-procedure:knee surgery` / `sha256:f858bc1d147108153a85001253e5b5e56b55a1c080fab2133a053aeacbd915f4` / `sha256:8b60572a3bc096e26e0678eda51c62d7cc31a7b6d50907ca606f424d62b7805f` / `surgical procedure`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `medical-procedure` / `medical-procedure:knee surgery` / `sha256:f858bc1d147108153a85001253e5b5e56b55a1c080fab2133a053aeacbd915f4` / `sha256:8633838896c19ebb40290e2815e957aeca8bf056ae76d5392dc9f7a9e6975a18` / `medical procedure`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:epilepsy` / `sha256:ad290e962fc9ae45736a60469b23f7eea0e33db5577069b8ed864a7c8a2c72c0` / `sha256:e2a39f92b8a616b325f87465af1ae3fee9f00e384615beb9291e60e196af79a4` / `central nervous system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:epilepsy` / `sha256:ad290e962fc9ae45736a60469b23f7eea0e33db5577069b8ed864a7c8a2c72c0` / `sha256:ca85cc67961efbec7dbda861832bd80c534ef29e77ece735452078f633200575` / `nervous system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:skin cancer` / `sha256:36ef3f3f7d030922ed4fae9aa4a45d1a1aa97f3ce7ff402dafcb1e89952e1acd` / `sha256:82736d5ad00eb1ab4638f8a4421b7d93a4260d23ed952b4453ca6e12df51729d` / `cancer`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:skin cancer` / `sha256:36ef3f3f7d030922ed4fae9aa4a45d1a1aa97f3ce7ff402dafcb1e89952e1acd` / `sha256:0cc5be82f6eaab0830ba2da5133f82a292b36ed4a1fd0a559394364cfc914b54` / `disease of cellular proliferation`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:epicondylitis` / `sha256:6852e0ac3353e6bac07ffa07570a97227279a71b78944db0b5d13f52d81d7b49` / `sha256:4c606a18e029e590f721187d6b63a47dbb87895aeb975d7626a03f21ed37b03a` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:lung cancer` / `sha256:b2de4dc1804d337e341d57e5b1ab2ae866b61be81999fb5f5ab9622f643b75cc` / `sha256:43f5d9e05e1e4b6caebaa37c04d6db8f0b1797f07f8a38a08a4797e61e245c21` / `cancer`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:lung cancer` / `sha256:b2de4dc1804d337e341d57e5b1ab2ae866b61be81999fb5f5ab9622f643b75cc` / `sha256:cdbaf99e35c2b5772912fae614758c62e301466a9d8426877cfe8dfa1c29767d` / `disease of cellular proliferation`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `drug` / `drug:growth hormone` / `sha256:8de6e6533fae698381d1810567a7cd7651aa0686c3de72f4341c72141e48617c` / `sha256:fd6bccf965191f17608e65f9952422bbe7ab9708521c33592ce469690a429b89` / `hormone`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:rheumatoid arthritis` / `sha256:7cdd9adf0b19c361e0948626e425410c3c4408a3cbda75e2ef7dfaf9571df6b4` / `sha256:c882d382dd9294524cda22ab6d8f9c8b41cdc3c6b7e9204084d9244ca9afe936` / `immune system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `medical-procedure` / `medical-procedure:knee surgery` / `sha256:6d949ec5c3089f675fd41b0f1e7f22662b594f86b98d1a4661229a10d43a3d32` / `sha256:d314a19f2f4fc089bf721778cabea12bfc3b4759f241de95a4ed6b4054d22950` / `orthopedic surgical procedures`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `medical-procedure` / `medical-procedure:knee surgery` / `sha256:6d949ec5c3089f675fd41b0f1e7f22662b594f86b98d1a4661229a10d43a3d32` / `sha256:a6d8460b85475f5280cfec692f1ff71ad2915a3e561e33f0e2c4cc03f9551f42` / `surgical procedure`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `medical-procedure` / `medical-procedure:knee surgery` / `sha256:6d949ec5c3089f675fd41b0f1e7f22662b594f86b98d1a4661229a10d43a3d32` / `sha256:13d432744e075a14c219ffc1dbb5c1aa7e90f22cff5a7ccfbe99abdd5d3c0c8a` / `medical procedure`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:a0dc2be0881b8fae9b8dc5cb8dcceb313f79ad32c64fe0ed043ce61f2a038442` / `sha256:3c3ad490fe5c017215972807a07db4f63e44109a703a5c4792280730a61b2eea` / `glucose metabolism disorder`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:a0dc2be0881b8fae9b8dc5cb8dcceb313f79ad32c64fe0ed043ce61f2a038442` / `sha256:30c525c6ba0335a6f6a8c60db9cf342ce8f9a48b58fffc2ec2757afdd9419f2a` / `clinical health state`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:a0dc2be0881b8fae9b8dc5cb8dcceb313f79ad32c64fe0ed043ce61f2a038442` / `sha256:0beb1b237725be272fd642e224df840cd669b34d590abf43ace03f2ee2a36960` / `metabolic disorder`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `LOC` / `LOC:springfield` / `sha256:394695ba16f2f0075cf77ecdadb644341608b2b20199662c4539e2496bb7ed2d` / `sha256:974171c256338f8f7109a5534817c0f35e6bb61f1629c7ff0a4ab221ebfddff8` / `a city in United States`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:arthritis` / `sha256:156d0f16b8d15cb2dd7d8b43ebe8eedecc380835861ee54dc5f4e6ccb92f7b2a` / `sha256:4b1b378fd43318f8cc5a2e42fa421139e20c04538df581b81c9281de1e15ad84` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:melanoma` / `sha256:e91994fa1a328137527df229adc0ca23fd9a7e50969d2471a331ec4025c954ef` / `sha256:008dc708ef3381ef1effdb422418c4a2e559cb6e8ab86f0f19c36a657091b801` / `cancer`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:skin cancer` / `sha256:824bb95b6f7acaa46a083b5ad2347667491695140fba2420394e5990f931b612` / `sha256:5e80520960a91ac8533ae1b1b7a13f8f7029b683d97ddeeeefa4d5f133917ea2` / `cancer`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:skin cancer` / `sha256:824bb95b6f7acaa46a083b5ad2347667491695140fba2420394e5990f931b612` / `sha256:dd1a2e80410111b140f8391a905c2a326e6ad78474be73ea4ee63406093d1c4e` / `disease of cellular proliferation`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:pancreatic cancer` / `sha256:269d0605fc9688685252fd45c1e0bc5c9e726513727a7771812d56e322a2f2c4` / `sha256:6aba4e59971e1d2222bbf2b2faacc0a876d76718dcb4c5b7e2705606d9f72814` / `cancer`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:pancreatic cancer` / `sha256:269d0605fc9688685252fd45c1e0bc5c9e726513727a7771812d56e322a2f2c4` / `sha256:280b6ff69b432657c0d0983fa7644b8d2a024520c867e90d1180d03540949946` / `disease of cellular proliferation`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:breast cancer` / `sha256:27487e98589d15d9e3ca188290c94313e2dcdfd6e48c4886a84c3d1f8f73a5d3` / `sha256:92a4ca2bbca87eed4005a94f58e02c8d801bd958eade55d6337178d227fadfed` / `cancer`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:breast cancer` / `sha256:27487e98589d15d9e3ca188290c94313e2dcdfd6e48c4886a84c3d1f8f73a5d3` / `sha256:fad2f7300591cc75e62c77846a9bce7d91ddafb9542d987b35bbe538263cc47c` / `disease of cellular proliferation`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:cancer` / `sha256:e490b12c435721459643d4282503a4ab8cf3c11180e08d99a73053911c1911e3` / `sha256:347889cea56fe231d8ef58815af1d5e699aab3af90b57e31fbac82c8f1ae214e` / `disease of cellular proliferation`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:prostate cancer` / `sha256:4e3ef2db6022da2d759d9d154d52e649e4c4b1543d654562446dd8003a1897c7` / `sha256:9c4eaa4af75172e1ce9358808fab34a25c08f931d63f1d5411374b1238f85644` / `cancer`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:cancer` / `sha256:5aa37ac5902eeadda750ddf3558bce9590ac67d55c734895108e9676f6ea5292` / `sha256:fd93c75c6396f01d7bfa26049d8d8dfe1483a82c454bc2dfb7350997ab1e6fc2` / `disease of cellular proliferation`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `LOC` / `LOC:mcmurray` / `sha256:d326cca6178c5bfa9403d7118ab73a26ac864ffc4c58464273450ae78a9edd86` / `sha256:96d96fa502a40a3687351de254a891616ce0689bf45e0f5a7e7f952a253dc953` / `a city in Pennsylvania`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `LOC` / `LOC:mcmurray` / `sha256:d326cca6178c5bfa9403d7118ab73a26ac864ffc4c58464273450ae78a9edd86` / `sha256:bb3059bf1001a7cfd4eb235307ae1fbebd2c3bf06f9b5d4a134c46ab2c1bfe6f` / `a city in United States`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `LOC` / `LOC:armenia` / `sha256:6ef29dbed8329fdeddea86c245f10d9cde77e1875657600388398a18c768113e` / `sha256:5e9e2784e17a86c7487e29667a808fde03ce600b9a99d0cac2b07e3b3b621802` / `a city in quindío department`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `LOC` / `LOC:armenia` / `sha256:6ef29dbed8329fdeddea86c245f10d9cde77e1875657600388398a18c768113e` / `sha256:788129c904f2eae7214b9683469d31b4eff58e5d86e4c37d63b49e322de095f6` / `a city in colombia`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `LOC` / `LOC:armenia` / `sha256:6ef29dbed8329fdeddea86c245f10d9cde77e1875657600388398a18c768113e` / `sha256:1feb49489cb6dc97b9a9807ee32942607df96a39f700b72f5df1ebb74434f823` / `a city in south america`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:rheumatoid arthritis` / `sha256:a64767529b65d5d741de2813d5f913f50ec0c4ecbc07773a33d74d24292e26fb` / `sha256:15c7e6b3e95b955c77749733fac4fbeccf755e46ed11646bb6ee92788c91a368` / `immune system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:epicondylitis` / `sha256:e53bd1b5adc4b810d87c8104dde21c33b62bce0662a57fd25149f9817ad6cc6d` / `sha256:d9521b9bb559abc54af137e47363995048de8525abfbf1b3a2660854386c948d` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:7c4c99a97ecce4dfe423966c1aba0c6047f53483ecd174424ffd39b54952ecc2` / `sha256:6f9db3bd668ef73b01b430427339c77afcf7b1b2128e54667907ac9df3e73501` / `glucose metabolism disorder`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:7c4c99a97ecce4dfe423966c1aba0c6047f53483ecd174424ffd39b54952ecc2` / `sha256:1644cbc185ca91fed07b86ea21a3f3c7291d465771ab3ad8c422087e6a45c374` / `clinical health state`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:7c4c99a97ecce4dfe423966c1aba0c6047f53483ecd174424ffd39b54952ecc2` / `sha256:329879cf803c3942c650016385e72de648e0b6a6f9cc46a2ef6769ab259ef531` / `metabolic disorder`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `LOC` / `LOC:mcmurray` / `sha256:cc335982a3bf6ee2915d5965bdab47c78ed77e159eb90963287c449f36e45705` / `sha256:2230752ad33d005a8c05dc78918c7c6b572ab7f29493cbeeff5700470487175a` / `a city in Pennsylvania`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `LOC` / `LOC:mcmurray` / `sha256:cc335982a3bf6ee2915d5965bdab47c78ed77e159eb90963287c449f36e45705` / `sha256:9106aa6bf949e0f343acbba0b0180045833cfe5f877b2f3b4d677f6ad0e749d0` / `a city in United States`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:patellofemoral pain syndrome` / `sha256:7f66de4e8d87f389a3c7ee62e3cbb90f68f7351614d4df55ca898242db7cb5d0` / `sha256:f48af994e3d18a6b71443223db921059597ff1b8e8781064c6549ad834e359d6` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:arthritis` / `sha256:f295d59c33a2dbe6acb2251750b2e2ebb7d8a682dbbdffd1cdcfb9d4abcfcd62` / `sha256:3a9c6a65aa5feea1c18dbc12c66d7af20e6057d02f0c08cb7974d83d3ede16ad` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `LOC` / `LOC:albania` / `sha256:9c9bc2947ea0e338af4ae1a6f1787f7465b5836598f7b875d0efbb8fe6a647ab` / `sha256:ed2ba0b1840735f5566954116fddea55d9cabc6d8190b8394d06e3372eb732a7` / `a city in la guajira department`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `LOC` / `LOC:albania` / `sha256:9c9bc2947ea0e338af4ae1a6f1787f7465b5836598f7b875d0efbb8fe6a647ab` / `sha256:0d871e3f0cc12a8ab88ba88ecb9074b76d24b61d2e6b5e4fe9b5795143c0970b` / `a city in colombia`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `LOC` / `LOC:albania` / `sha256:9c9bc2947ea0e338af4ae1a6f1787f7465b5836598f7b875d0efbb8fe6a647ab` / `sha256:785424167766dc6321a45572d318fc9e7ccb41e0bc3bb9181e8c9ce475cc85e4` / `a city in south america`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:arthritis` / `sha256:f06215c1ba19cb614ebfb5a25d2a355b122bf42da81b5d65d64c838b3742d46f` / `sha256:61b87a0ad4a0cb6a4a6f19be6b9a946166b82a96d8e597b2c807b84be5322ea9` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:e85821eeb80b4701b79cb845dbd416f8ea803950ae38bd61c79eb66141284c98` / `sha256:a944534b2eb0924af59ce69d80a86ec211dfd4d98b445bd151214a48aaf10194` / `glucose metabolism disorder`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:e85821eeb80b4701b79cb845dbd416f8ea803950ae38bd61c79eb66141284c98` / `sha256:d891d2b74c2beabfdd70775ea26c0989b98d2b3d98eb628f9d1eb21fefabf47a` / `clinical health state`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:e85821eeb80b4701b79cb845dbd416f8ea803950ae38bd61c79eb66141284c98` / `sha256:dd1b562fbac8e1ec13a35d2c9f4335a8bdc77ffe72a5a2fcdd0cad46fb8d5a0f` / `metabolic disorder`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:smoker` / `sha256:755a2c8d0cb4d689c6049c1f678f08d85a0214fbb2a0d0911a5998fbfd856556` / `sha256:881d2945da4ff8e4ffd99399457f971dec486ccafd108199aa96b01dd04d1a0b` / `clinical health state`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:arthritis` / `sha256:bf0d3cd3036f58c7dddfb68a6177234094e3535cd54b88b817323f94640ab1fb` / `sha256:9863091d5d9ea02e849a4e347ac88a7bf37dcf270f6240e2aa33490a39d64626` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:lower respiratory infections` / `sha256:45750f796493d39d1476ac915b007a2223f8131f5e44505d1763941f944497dc` / `sha256:09676adee9af80b5ce1bb9a6cf33a4f0afa69eb4bbc7997521e576042f9eb6d8` / `lower respiratory tract disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:arthritis` / `sha256:621cc222c759049f989bb4add19ddb175c599b2d0f28be4c37df90326a3e5571` / `sha256:cdf2aa7781af53b53bd5bead0695cffa795ed6675458ad548ea323b44107a2c5` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:arthritis` / `sha256:1b2a729c4dba5811c31e10eb3477d3f5585632fb255f59c16afae7200eecad7a` / `sha256:a270d8a1d213457538cd3cae2e60d457af20a2e81ce7cd56252181248d596710` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:aef01c3dbfba00e999f9bff7c76eaad1406f6acea1200dca2599bd87f9a9f771` / `sha256:a978c4420c60e343c4ce64b623d18b96321a9fb215a5ae2fd8ba69f4e616a153` / `glucose metabolism disorder`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:aef01c3dbfba00e999f9bff7c76eaad1406f6acea1200dca2599bd87f9a9f771` / `sha256:0390a56061fe1feb07aa8c07538e349c9cf45e853b9ffe4ebaeb96c560335977` / `clinical health state`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:aef01c3dbfba00e999f9bff7c76eaad1406f6acea1200dca2599bd87f9a9f771` / `sha256:3c47bac255e3380bf1372c31beb7328954820134bdc4cbecf7352c73d4b45af2` / `metabolic disorder`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:1f57d690b139d4be350fa653bf6e6f6d9885f40c9551ac5176946d5dc3704b1f` / `sha256:359c63b699504a548d829476a9c0cc650156c731ea044b8fc0a8afb3da42d31b` / `glucose metabolism disorder`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:1f57d690b139d4be350fa653bf6e6f6d9885f40c9551ac5176946d5dc3704b1f` / `sha256:0c943ea11e116660bd04f2e8a7bd89b9f634dd80dc0f6e0e953c997bf49e9840` / `clinical health state`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `None` / `sha256:1f57d690b139d4be350fa653bf6e6f6d9885f40c9551ac5176946d5dc3704b1f` / `sha256:5ca82759b288305a88afbfb12ccf36d0d8cd6dfe8f3298ab5bc5201029ea4ad1` / `metabolic disorder`: accepted_status, explicit_count, matched_profile, nondefault_provenance, status_evidence.
- `health-condition` / `health-condition:arthritis` / `sha256:7d1e7fa8e8691ce4c50268ac9b30f127ab90c7d9218015dcc77c6c27492811e7` / `sha256:b370e78e5dcf1568931fde63f2fcad2cd1a3b153c8b107384448376041129fc8` / `musculoskeletal system disease`: accepted_status, explicit_count, nondefault_provenance, status_evidence.
