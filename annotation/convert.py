# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "polars==1.32.3",
#     "pyyaml==6.0.2",
#     "spacy==3.8.7",
# ]
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from glob import glob
    from pathlib import Path
    import json
    import yaml
    import polars as pl
    import spacy
    from os.path import exists
    from spacy.cli.train import train
    from spacy.cli.package import package
    from spacy.cli.download import download
    return Path, exists, glob, mo, pl, yaml


@app.cell
def _(mo):
    mo.md(r"""# Converting ner-annotator jsons to Spacy Docs""")
    return


@app.cell
def _(Path):
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    return (NOTEBOOK_DIR,)


@app.cell
def _(NOTEBOOK_DIR, mo):
    annotation_files = mo.ui.text(value=f'{NOTEBOOK_DIR}/*.json', label = 'Path to annotation files')
    annotation_files
    return (annotation_files,)


@app.cell
def _(annotation_files, glob):
    datafiles = glob(annotation_files.value)
    return (datafiles,)


@app.function
def get_data(datafiles):
    import json
    data = []
    for file in sorted(datafiles):
        with open(file, 'r') as fp:
            j = json.load(fp)
        for annotation in j['annotations']:
            if annotation[0] == '' or annotation[0] == '\n':
                continue
            d = {}
            d['text'] = annotation[0]
            d['entities'] = [(int(e[0]), int(e[1]), e[2]) for e in annotation[1]['entities']]
            data.append(d)
    return data


@app.cell
def _():
    import random

    def distribute_ids(K, percentages):
        # Check if the percentages sum to 1 (with a tolerance for floating-point errors)
        #if abs(sum(percentages) - 1) > 1e-6:
        #    raise ValueError("The percentages must sum to 1.")
        assert sum(percentages) == 1.0

        # Number of sublists (N) is the length of the percentages list
        N = len(percentages)

        # Calculate the number of integers that should go into each sublist
        sublist_sizes = [int(K * p) for p in percentages]

        # Handle rounding issues in the sublist sizes (because K might not be perfectly divisible)
        total_assigned = sum(sublist_sizes)
        if total_assigned < K:
            sublist_sizes[-1] += K - total_assigned  # Add any remaining numbers to the last sublist

        # List to hold the sublists
        sublists = [[] for _ in range(N)]

        # Create a list of numbers from 1 to K
        numbers = list(range(K))

        # Shuffle the numbers randomly
        random.shuffle(numbers)

        # Distribute the shuffled numbers into the sublists
        start_idx = 0
        for i in range(N):
            end_idx = start_idx + sublist_sizes[i]
            sublists[i] = numbers[start_idx:end_idx]
            start_idx = end_idx

        return sublists
    return (distribute_ids,)


@app.cell
def _(datafiles):
    all_data = get_data(datafiles)
    return (all_data,)


@app.cell
def _(NOTEBOOK_DIR, exists, mo):
    distribution_path = f'{NOTEBOOK_DIR}/distribution.yaml'
    if exists(distribution_path):
        existing_distribution = True
    else:
        existing_distribution = False
    old_dist_ui = mo.ui.switch(label='Use existing distribution', value=existing_distribution)
    old_dist_ui
    return distribution_path, old_dist_ui


@app.cell
def _(distribution_path, yaml):
    def get_dist():
        with open(distribution_path, 'r') as f:
            dist = yaml.safe_load(f)
            return dist['training'], dist['dev'], dist['test']
    return (get_dist,)


@app.cell
def _(
    all_data,
    distribute_ids,
    distribution_path,
    exists,
    get_dist,
    mo,
    old_dist_ui,
    yaml,
):
    if old_dist_ui.value and exists(distribution_path):
        mo.output.append(mo.md(f'Using distribution in {distribution_path}'))
        training, dev, test = get_dist()
    else:
        mo.output.append(mo.md(f'Generating new distribution and writing to {distribution_path}'))
        training, dev, test = distribute_ids(len(all_data), [0.8,0.1,0.1])
        with open(distribution_path, 'w') as _f:
            yaml.dump({'training': training, 'dev': dev, 'test': test}, _f)
    return dev, test, training


@app.cell
def _(all_data, dev, test, training):
    training_data = [all_data[i] for i in training]
    dev_data = [all_data[i] for i in dev]
    test_data = [all_data[i] for i in test]
    return dev_data, test_data, training_data


@app.cell
def _(mo):
    mo.md(r"""### Entity Distribution""")
    return


@app.function
def get_entity_distribution(data):
    from collections import Counter
    d = {}
    for datum in data:
        entities = [e[2] for e in datum['entities']]
        counts = Counter(entities)
        for type, count in counts.items():
            d[type] = d.get(type, 0) + count

    p = {}
    total = sum(d.values())
    for t, v in d.items():
        p[t] = 100.0 * v / total

    return d, p


@app.cell
def _(pl):
    def get_df(**dicts):
        all_keys = sorted(set().union(*[d.keys() for d in dicts.values()]))

        data = {'row': all_keys}
        for label, d in dicts.items():
            data[label] = [d.get(k, 0) for k in all_keys]

        df = pl.DataFrame(data, strict=False)

        return df
    return (get_df,)


@app.cell
def _(all_data, dev_data, get_df, mo, test_data, training_data):
    all_dist = get_entity_distribution(all_data)
    training_dist = get_entity_distribution(training_data)
    dev_dist = get_entity_distribution(dev_data)
    test_dist = get_entity_distribution(test_data)
    mo.vstack([
        mo.md('**Count for each entity for the combined dataset and the training, dev, and test sets**'),
        get_df(all=all_dist[0], training=training_dist[0], dev=dev_dist[0], test=test_dist[0]),
        mo.md('**Percentage of each entity for the combined dataset and the training, dev, and test sets**'),
        get_df(all=all_dist[1], training=training_dist[1], dev=dev_dist[1], test=test_dist[1])
    ])
    return


@app.cell
def _(mo):
    write_button = mo.ui.run_button(label="Write to files")
    write_button
    return (write_button,)


@app.cell
def _(mo):
    mo.md(r"""### Writing to Spacy File""")
    return


@app.cell
def _(NOTEBOOK_DIR, mo):
    def write_to_spacy_doc(**data_list):
        from spacy import blank
        from spacy.tokens import DocBin
        from tqdm import tqdm
        from spacy.util import filter_spans

        for fname, data in data_list.items():
            mo.output.append(mo.md(f'Creating spacy DocBin object for {fname}'))
            nlp = blank('en')
            nlp.add_pipe('sentencizer')
            doc_bin = DocBin()
            for datum in tqdm(data):
                text = datum['text']
                labels = datum['entities']
                doc = nlp.make_doc(text)
                ents = []
                for start, end, label in labels:
                    n = len(text)
                    if start < 0 or end < 0 or start > n or end > n:
                        print(f"Clamping out-of-bounds entity: ({start}, {end}, {label}) for text length {n}")
                        start = max(0, min(start, n))
                        end = max(0, min(end, n))

                    while text[start] in [' ', '\n', '\r']:
                        start = start + 1
                    while text[end-1] in [' ', '\n', '\r']:
                        end = end - 1
                    span_text = text[start:end]
                    if text[start] == ' ' or text[end-1] == ' ' or span_text != span_text.strip():
                        print(f'entity ({start}, {end}, {label}) begins/ends with whitespace: <{text[start:end]}>')
                    span = doc.char_span(start, end, label=label, alignment_mode='expand')
                    if span is None:
                        print(f'skipping entity: ({start}, {end}, {label})')
                    else:
                        ents.append(span)
                filtered_ents = filter_spans(ents)
                doc.ents = filtered_ents
                doc_bin.add(doc)

            mo.output.append(mo.md(f'Writing to {NOTEBOOK_DIR}/{fname}.spacy'))
            doc_bin.to_disk(f'{NOTEBOOK_DIR}/{fname}.spacy')
    return (write_to_spacy_doc,)


@app.cell
def _(
    all_data,
    dev_data,
    test_data,
    training_data,
    write_button,
    write_to_spacy_doc,
):
    if write_button.value:
        write_to_spacy_doc(all=all_data, training=training_data, dev=dev_data, test=test_data)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
