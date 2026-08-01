# Privacy Estimator Computational Graph Design

## Objective

Replace the coarse inline-SVG architecture figure in `docs/html/privacy-estimator.html` with a component-level computational graph that uses the visual and routing system of `docs/html/interactive-ranker-v2.html` FIG M1.

The figure documents the implemented model in `src/cloak/train/ml_k_anonymity.py` on branch `codex/privacy-relation-counterexamples`. It does not propose a replacement architecture and does not change the surrounding abstract or model-overview prose.

## Rendering architecture

Use the existing hybrid diagram system:

- semantic HTML nodes establish a rigid CSS-grid layout;
- existing `ranker-model-diagram` classes encode node shapes and component roles;
- `@viz-js/viz` runs Graphviz in the browser;
- `docs/html/js/ranker-model-diagram.js` measures the HTML nodes and routes orthogonal edges around them with pinned `neato` positions;
- the HTML remains readable when edge routing fails, with the existing status message explaining the failure.

The page imports the same pinned Viz.js version and shared router as the reference page. No new JavaScript, stylesheet, or package dependency is introduced.

## Graph organization

The rigid graph contains two vertically stacked modules because relation and count are separate forward passes that share checkpoint weights rather than one joint batch.

### Relation module

Show these operations as distinct nodes:

1. runtime type, detected source, candidate substitution, and marked occurrence excerpt;
2. metadata and context serialization as tokenizer sequence A and sequence B;
3. pair tokenization with second-sequence-only truncation to 256 tokens;
4. length-bucketed dynamic padding, bounded by 4,096 tokens and 16 examples;
5. `input_ids`, `attention_mask`, `token_type_ids` when available, and `special_tokens_mask`, each shaped `[B,L]`;
6. ClinicalMosaic embeddings and encoder layers 0 through 7, frozen;
7. encoder layers 8 through 11, trainable at relation time;
8. final hidden states `H_rel [B,L,768]`;
9. boolean pooling mask `attention_mask AND NOT special_tokens_mask`;
10. masked sum divided by the count of poolable tokens, yielding `z_rel [B,768]` in float32;
11. `LayerNorm(768)`;
12. `Linear(768,128)`;
13. GELU;
14. `Dropout(0.1)`;
15. `Linear(128,4)`;
16. four relation logits and softmax-at-inference labels;
17. balanced four-class cross-entropy, with sampling balanced by label, bundle, and rows within each bundle;
18. AdamW parameter groups with encoder learning rate `2e-5`, head learning rate `2e-4`, weight decay `0.01`, and global gradient clipping at `1.0`.

### Count module

Show these operations as distinct nodes:

1. runtime type and candidate substitution only;
2. candidate-only serialization and tokenization to at most 256 tokens;
3. relation-adapted ClinicalMosaic with the complete encoder frozen;
4. final hidden states `H_count [B,L,768]` and the same masked-mean operation;
5. cached CPU features `z_count [N,768]`;
6. `LayerNorm(768)`;
7. `Linear(768,64)`;
8. GELU;
9. `Linear(64,1)`;
10. standardized prediction `y_hat_std`;
11. inverse target transform `y_hat_logK = y_hat_std * sigma_train + mu_train` for reporting;
12. formal universe target `log(count_target)`, standardized by the training mean and population standard deviation;
13. Smooth-L1 regression on sampled formal universes;
14. formal narrow and broad universe pairs, scored through the same count head;
15. difference `Delta = y_hat_broad - y_hat_narrow` and pairwise loss `softplus(-Delta)`;
16. total count loss `L_count = L_reg + 1.0 * L_pair`;
17. AdamW over the count head only at learning rate `3e-4`, weight decay `0.01`, and gradient clipping at `1.0`.

A dashed weight-tie edge connects the relation encoder output checkpoint to the frozen count encoder, making the stage order explicit without implying a joint forward pass.

## Shape and color semantics

Reuse the reference diagram semantics:

- rounded rectangles: inputs, serialization, batching, and composite modules;
- green rectangles: encoders, cached tensors, and tensor states;
- rust trapezoids: linear projections;
- green hexagons: GELU activations;
- gray square-corner rectangles: masks, normalization, dropout, and frozen transforms;
- circles: arithmetic operations such as masked mean, standardization, subtraction, and loss addition;
- pills: logits, predicted log count, and final losses;
- dashed node borders: trainable components;
- dashed edges: gradients, checkpoint transfer, or target-only supervision;
- solid orthogonal edges: forward tensor flow.

## Responsive behavior

The diagram keeps a minimum width sufficient for legible 10 to 13 pixel monospace labels. The report frame uses horizontal overflow at narrow viewports, matching the reference page. Desktop rendering must expose the complete graph without clipped nodes or crossed node interiors. At 768 pixels, prose reflows and the diagram remains horizontally scrollable.

## Validation

Validation must establish all of the following:

- HTML parses and contains no placeholders;
- exactly one Graphviz DOT source exists in the page;
- every DOT endpoint has one matching `data-node-id` element;
- every required component and tensor shape is present;
- the page imports Viz.js and `ranker-model-diagram.js` exactly once;
- the shared router reports `data-renderer="grid-neato"` in a rendered browser DOM;
- desktop and narrow screenshots show rigid alignment, distinct shapes, readable labels, and orthogonal arrows that do not cross node interiors;
- `docs/html/js/nav.js` remains syntactically valid;
- only the report, design record, and implementation plan are committed.
