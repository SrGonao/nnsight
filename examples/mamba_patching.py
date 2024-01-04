import sys
sys.path.append('/mnt/c/Users/g-spa/Desktop/nnsight/src/')
from nnsight.tracing.Proxy import Proxy
from nnsight import util
from nnsight.models.Mamba import Mamba

import torch

# Make sure you're on the Mamba nnsight branch: https://github.com/JadenFiotto-Kaufman/nnsight/tree/mamba
# Also need to install mamba specific packages:
# pip install causal-conv1d>=1.1.0
# pip install mamba-ssm
model = Mamba("state-spaces/mamba-130M", device="cuda")

# Set clean prompt.
clean_prompt = "After John and Mary went to the store, Mary gave a bottle of milk to"
# Set corrupted prompt with pronouns swapped.
corrupted_prompt = (
    "After John and Mary went to the store, John gave a bottle of milk to"
)

# Get the vocabulary indices for the correct and incorrect prediction.
correct_index = model.tokenizer(" John")["input_ids"][0]
incorrect_index = model.tokenizer(" Mary")["input_ids"][0]

# Indicate we want to run the model.
with model.forward(validate=False) as generator:
    # Indicate we want to enter some input into the model.
    # Run clean prompt though in order to grab the clean hidden states.
    with generator.invoke(clean_prompt, scan=False) as invoker:
        clean_tokens = invoker.input["input_ids"][0]

        # Grab hidden states as output of each layer module.
        clean_hs = [
            model.backbone.layers[layer_idx].output
            for layer_idx in range(len(model.backbone.layers))
        ]

        # Grab clean logits
        clean_logits = model.lm_head.output

        # Compute logit diff between correct and incorrect indices.
        clean_logit_diff = (
            clean_logits[0, -1, correct_index] - clean_logits[0, -1, incorrect_index]
        ).save()

    # Run a single corrupted prompt to get logit diff of corrupted.
    with generator.invoke(corrupted_prompt, scan=False) as invoker:
        corrupted_logits = model.lm_head.output

        corrupted_logit_diff = (
            corrupted_logits[0, -1, correct_index]
            - corrupted_logits[0, -1, incorrect_index]
        ).save()

    ioi_patching_results = []

    # Iterate through each layer.
    for layer_idx in range(len(model.backbone.layers)):
        _ioi_patching_results = []

        # Iterate through each token.
        for token_idx in range(len(clean_tokens)):
            with generator.invoke(corrupted_prompt, scan=False) as invoker:
                # Patch in hidden states from clean run into corrupted at given layer and token.
                model.backbone.layers[layer_idx].output[0][:, token_idx] = clean_hs[
                    layer_idx
                ][0][:, token_idx]
                # Output of each layer is a tuple of (hidden_states, residual). Need to patch both.
                model.backbone.layers[layer_idx].output[1][:, token_idx] = clean_hs[
                    layer_idx
                ][1][:, token_idx]

                patched_logits = model.lm_head.output

                patched_logit_diff = (
                    patched_logits[0, -1, correct_index]
                    - patched_logits[0, -1, incorrect_index]
                )

                # Compute improvement in logit diff.
                patched_result = (patched_logit_diff - corrupted_logit_diff) / (
                    clean_logit_diff - corrupted_logit_diff
                )

                _ioi_patching_results.append(patched_result.save())

        ioi_patching_results.append(_ioi_patching_results)


print(f"Clean logit difference: {clean_logit_diff.value:.3f}")
print(f"Corrupted logit difference: {corrupted_logit_diff.value:.3f}")

# Convert proxies to their value.
ioi_patching_results = util.apply(ioi_patching_results, lambda x: x.value.item(), Proxy)

import plotly.express as px

clean_tokens = [model.tokenizer.decode(token) for token in clean_tokens]
token_labels = [f"{token}_{index}" for index, token in enumerate(clean_tokens)]

fig = px.imshow(
    ioi_patching_results,
    color_continuous_midpoint=0.0,
    color_continuous_scale="RdBu",
    labels={"x": "Position", "y": "Layer"},
    x=token_labels,
    title="Normalized Logit Difference After Patching Residual Stream on the IOI Task",
)

#open the figure in a browser
fig.show()
#fig.write_image("patching.png")
