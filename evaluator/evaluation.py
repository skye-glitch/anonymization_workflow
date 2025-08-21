# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "matplotlib==3.10.5",
#     "pandas==2.3.1",
#     "presidio-evaluator==0.2.4",
#     "scikit-learn==1.7.1",
#     "itables==2.4.4",
#     "pip==25.2",
#     "presidio-analyzer==2.2.359",
#     "requests==2.32.5",
#     "spacy==3.8.7",
#     "spacy-curated-transformers==0.3.1",
#     "spacy-transformers==1.3.9",
#     "plotly==5.24.1",
#     "ipython==9.4.0",
#     "kaleido==0.2.1",
#     # "en_helpdesktickets==0.1",
# ]
# [tool.uv.sources]
# # en_helpdesktickets = { path = "../training/packages/en_helpdesktickets-0.1/dist/en_helpdesktickets-0.1-py3-none-any.whl", editable = true }
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Evaluate Analyzer""")
    return


@app.cell
def _(mo):
    def clear_results(new):
        df = None
    new_experiment_switch = mo.ui.switch(label='Run new experiment', on_change=clear_results)
    return (new_experiment_switch,)


@app.cell
def _(new_experiment_switch):
    new_experiment_switch
    return


@app.cell
def _():
    from pathlib import Path
    import sys
    import pandas as pd
    import json
    import matplotlib.pyplot as plt

    from collections import Counter
    from glob import glob
    from typing import Dict, List
    from pprint import pprint
    from datetime import datetime
    from os import makedirs, getcwd
    from os.path import basename
    from shutil import move
    from sklearn.metrics import precision_score, recall_score, f1_score, fbeta_score

    from presidio_evaluator import InputSample, Span
    from presidio_evaluator.evaluation import Evaluator, ModelError, Plotter
    from presidio_evaluator.experiment_tracking import get_experiment_tracker
    from presidio_evaluator.models import PresidioAnalyzerWrapper

    pd.set_option('display.max_rows', None)
    return (
        Counter,
        Dict,
        Evaluator,
        InputSample,
        List,
        Path,
        Plotter,
        PresidioAnalyzerWrapper,
        Span,
        basename,
        datetime,
        f1_score,
        fbeta_score,
        get_experiment_tracker,
        getcwd,
        glob,
        json,
        makedirs,
        move,
        pd,
        plt,
        pprint,
        precision_score,
        recall_score,
        sys,
    )


@app.cell
def _(Path):
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    return (NOTEBOOK_DIR,)


@app.cell
def _(NOTEBOOK_DIR, mo, new_experiment_switch, sys):
    analyzer = None
    if new_experiment_switch.value:
        mo.output.append(mo.md(r"""## Getting analyzer"""))
        sys.path.append(f'{NOTEBOOK_DIR}/../analyzer')
        import analyzer
    return (analyzer,)


@app.cell
def _(analyzer):
    analyzer_state = analyzer.app.run()
    return (analyzer_state,)


@app.cell
def _(analyzer_state):
    myAnalyzer = analyzer_state[1]['analyzer']
    nlp_engine = analyzer_state[1]['nlp_engine']
    token_model_version = analyzer_state[1]['configuration']['models'][0]['model_name']
    return myAnalyzer, nlp_engine, token_model_version


@app.cell
def _(Span, nlp_engine, token_model_version):
    def docbin_to_dataset(doc_path):
        from presidio_evaluator import InputSample
        from spacy.tokens import DocBin
        sentencizer = nlp_engine.nlp['en'].get_pipe('sentencizer')
        dataset = []
        docbin = DocBin().from_disk(doc_path)
        docs = list(docbin.get_docs(nlp_engine.nlp['en'].vocab))
        dataset = []
        for doc in docs:
            doc = sentencizer(doc)
            text = doc.text
            if text.strip() == '':
                continue
            spans = [Span(start_position=ent.start_char, end_position=ent.end_char, entity_type=ent.label_, entity_value=ent.text) for ent in doc.ents]
            dataset.append(InputSample(full_text=text, spans = spans, tokens=doc, create_tags_from_span=True, token_model_version=token_model_version))
        return dataset
    return (docbin_to_dataset,)


@app.cell
def _(NOTEBOOK_DIR, mo, new_experiment_switch):
    dataset_path = None
    if new_experiment_switch.value:
        dataset_path = mo.ui.text(value=f'{NOTEBOOK_DIR}/../annotation/dev.spacy', label='Annotated dataset path: ')
    dataset_path
    return (dataset_path,)


@app.cell
def _(dataset_path, docbin_to_dataset, mo):
    mo.output.append(mo.md(r"""## Getting Annotations"""))
    dataset = docbin_to_dataset(dataset_path.value)
    return (dataset,)


@app.cell(hide_code=True)
def _(mo, new_experiment_switch):
    if new_experiment_switch.value:
        mo.output.append(mo.md(r"""## Building Evaluator"""))
    return


@app.cell
def _(Counter, Dict, InputSample, List):
    def get_entity_counts(dataset: List[InputSample]) -> Dict:
        """Return a dictionary with counter per entity type."""
        entity_counter = Counter()
        for sample in dataset:
            for tag in sample.tags:
                entity_counter[tag] = entity_counter[tag] + 1
        return entity_counter
    return (get_entity_counts,)


@app.cell
def _(Evaluator, PresidioAnalyzerWrapper, dataset, get_entity_counts, pprint):
    entities_mapping = PresidioAnalyzerWrapper.presidio_entities_map
    entities_mapping['USERNAME'] = 'USERNAME'
    entities_mapping['PROJECT_NUMBER'] = 'PROJECT_NUMBER'
    entities_mapping['PUBLIC_URL_IP'] = 'PUBLIC_URL_IP'
    entities_mapping['SECRET_URL_IP'] = 'SECRET_URL_IP'
    entities_mapping['PORT_NUMBER'] = 'PORT_NUMBER'
    entities_mapping['CLIENT_NAME'] = 'PERSON'
    entities_mapping['CLIENT_COMPANY_NAME'] = 'CLIENT_COMPANY_NAME'
    entities_mapping['PASSWORD_CODE'] = 'PASSWORD_CODE'
    entities_mapping['ETHNICITY'] = 'NRP'
    entities_mapping['ADDRESS'] = 'LOCATION'
    entities_mapping['PUBLIC_ADDRESS'] = 'PUBLIC_ADDRESS'
    entities_mapping['SSN'] = 'US_SSN'
    print("Using this mapping between the dataset and Presidio's entities:")
    pprint(entities_mapping, compact=True)
    dataset_aligned = Evaluator.align_entity_types(dataset, entities_mapping=entities_mapping)
    new_entity_counts = get_entity_counts(dataset_aligned)
    print('\nCount per entity after alignment:')
    pprint(new_entity_counts.most_common(), compact=True)
    dataset_entities = list(new_entity_counts.values())
    return dataset_aligned, entities_mapping


@app.cell
def _(
    Evaluator,
    dataset,
    entities_mapping,
    get_experiment_tracker,
    json,
    mo,
    myAnalyzer,
    token_model_version,
):
    mo.output.append(mo.md(r"""## Running Experiment"""))
    experiment = get_experiment_tracker()
    evaluator = Evaluator(model=myAnalyzer)
    params = {'dataset_name': 'my_dataset', 'model_name': token_model_version}
    params.update(evaluator.model.to_log())
    experiment.log_parameters(params)
    experiment.log_dataset_hash(dataset)
    experiment.log_parameter('entity_mappings', json.dumps(entities_mapping))
    return evaluator, experiment


@app.cell
def _(dataset_aligned, datetime, evaluator, experiment, mo):
    evaluation_results = evaluator.evaluate_all(dataset_aligned)
    results = evaluator.calculate_score(evaluation_results)
    experiment.log_metrics(results.to_log())
    entities, confmatrix = results.to_confusion_matrix()
    experiment.log_confusion_matrix(matrix=confmatrix, labels=entities)
    experiment_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment.end()
    mo.output.append(mo.md(r"""## Experiment Post-Processing"""))
    return (
        confmatrix,
        entities,
        evaluation_results,
        experiment_timestamp,
        results,
    )


@app.cell
def _(NOTEBOOK_DIR, basename, glob, mo, new_experiment_switch):
    csv_file = None
    if not new_experiment_switch.value:
        csv_file = mo.ui.dropdown(options=sorted([basename(csv) for csv in glob(f'{NOTEBOOK_DIR}/results/*.csv')], reverse=True), label = 'Result CSV file')
    csv_file
    return (csv_file,)


@app.cell
def _(NOTEBOOK_DIR, getcwd, glob, makedirs, move):
    def move_results():
        run_dir = getcwd()
        results_dir = f'{NOTEBOOK_DIR}/results'
        makedirs(results_dir, exist_ok=True)
        result_files = glob('*.csv') + glob('*.json')
        for file in result_files:
            move(f'{run_dir}/{file}', f'{results_dir}/{file}')
    return (move_results,)


@app.cell
def _(
    NOTEBOOK_DIR,
    csv_file,
    evaluation_results,
    evaluator,
    experiment_timestamp,
    mo,
    move_results,
    new_experiment_switch,
    pd,
):
    _msg = ''
    if new_experiment_switch.value:
        df = evaluator.get_results_dataframe(evaluation_results)
        df.to_csv(f'experiment_{experiment_timestamp}.csv')
        _msg += f'Wrote results to experiment_{experiment_timestamp}.csv'
    else:
        if csv_file.value:
            _msg += f'Reading in dataframe from **{csv_file.value}**<br>'
            try:
                df = pd.read_csv(f'{NOTEBOOK_DIR}/results/{csv_file.value}')
                _msg += f'<span style="color: green">Succesfully loaded!</span>'
            except FileNotFoundError:
                _msg += f'<span style="color: red">  Warning</span>: file **{csv_file.value}** not found!'
    move_results()
    mo.output.append(mo.md(_msg))
    return (df,)


@app.cell
def _():
    import plotly.graph_objects as go
    from contextlib import contextmanager
    from IPython.display import display

    @contextmanager
    def capture_and_display_plotly():
        """Capture any Plotly figure shown inside the block and display it inline."""
        captured = {}

        original_show = go.Figure.show

        def capture_show(self, *args, **kwargs):
            captured['fig'] = self
            display(self)  # Automatically display inline
            return self  # Do not open a separate window

        # Patch
        go.Figure.show = capture_show

        try:
            yield captured
        finally:
            go.Figure.show = original_show  # Restore original method
    return (capture_and_display_plotly,)


@app.cell
def _(
    Plotter,
    capture_and_display_plotly,
    confmatrix,
    entities,
    evaluator,
    mo,
    results,
):
    mo.output.append(mo.md(r"""### Plots"""))

    plotter = Plotter(results=results, output_folder='plots', model_name=evaluator.model.name, save_as='png', beta=2)
    with capture_and_display_plotly():
        plotter.plot_confusion_matrix(entities=entities, confmatrix=confmatrix)
    return (plotter,)


@app.cell
def _(capture_and_display_plotly, plotter):
    with capture_and_display_plotly():
        plotter.plot_scores()
    return


@app.cell
def _(pprint, results):
    pprint({'PII F': results.pii_f, 'PII recall': results.pii_recall, 'PII precision': results.pii_precision})
    return


@app.cell
def _(NOTEBOOK_DIR, experiment_timestamp, getcwd, move):
    def move_plots():
        run_dir = getcwd()
        results_dir = f'{NOTEBOOK_DIR}/plots_{experiment_timestamp}'
        move(f'{run_dir}/plots', results_dir)
    return (move_plots,)


@app.cell
def _(NOTEBOOK_DIR, glob, mo):
    def display_plots(csv_file):
        timestamp = csv_file.split('.')[0].split('_')[1]
        plot_dir = f'{NOTEBOOK_DIR}/plots_{timestamp}'
        images = []
        for imgfile in glob(f'{plot_dir}/*.png'):
            images.append(mo.image(src=imgfile))
        mo.output.append(mo.vstack(images))
    return (display_plots,)


@app.cell
def _(csv_file, display_plots, move_plots, new_experiment_switch):
    if new_experiment_switch.value:
        move_plots()
    else:
        display_plots(csv_file.value)
    return


@app.cell
def _(df, mapping, mo):
    # Dataframe adjustments
    mo.output.append(mo.md(r"""### Metrics"""))
    total_tokens = df.shape[0]
    dataframe = df[(df['annotation'] != 'O') | (df['prediction'] != 'O')]
    dataframe['annotation'] = dataframe['annotation'].replace(mapping)
    dataframe['prediction'] = dataframe['prediction'].replace(mapping)

    incorrect = dataframe[(dataframe['annotation'] != dataframe['prediction']) & (dataframe['token'].apply(lambda x: any(c.isalnum() for c in str(x))))]
    correct = dataframe[(dataframe['annotation'] == dataframe['prediction']) & (dataframe['token'].apply(lambda x: any(c.isalnum() for c in str(x))))]
    return dataframe, incorrect, total_tokens


@app.cell
def _():
    mapping = {'ADDRESS': 'LOCATION', 
               'CLIENT_NAME': 'PERSON'}
    return (mapping,)


@app.cell
def _(f1_score, fbeta_score, mo, precision_score, recall_score):
    def get_metrics(df, total_tokens):
        precision = precision_score(df['annotation'], df['prediction'], average='weighted', zero_division=0.0)
        recall = recall_score(df['annotation'], df['prediction'], average='weighted', zero_division=0.0)
        f1 = f1_score(df['annotation'], df['prediction'], average='weighted', zero_division=0.0)
        f2 = fbeta_score(df['annotation'], df['prediction'], average='weighted', zero_division=0.0, beta=2)

        correct = df[(df['annotation'] == df['prediction']) & (df['token'].apply(lambda x: any(c.isalnum() for c in str(x))))]
        incorrect = df[(df['annotation'] != df['prediction']) & (df['token'].apply(lambda x: any(c.isalnum() for c in str(x))))]

        metrics = [
            {'Metric': 'Total tokens', 'Value': total_tokens},
            {'Metric': 'Total (tagged) tokens', 'Value': df.shape[0]},
            {'Metric': 'Correct predictions', 'Value': correct.shape[0]},
            {'Metric': 'Incorrect predictions', 'Value': incorrect.shape[0]},
            {'Metric': 'Precision', 'Value': f'{precision:.4f}'},
            {'Metric': 'Recall', 'Value': f'{recall:4f}'}, 
            {'Metric': 'F1 Score', 'Value': f'{f1:4f}'},
            {'Metric': 'F2 Score', 'Value': f'{f2:4f}'}
        ]
        return mo.ui.table(metrics)
    return (get_metrics,)


@app.cell
def _(dataframe, get_metrics, total_tokens):
    get_metrics(dataframe, total_tokens)
    return


@app.cell
def _(incorrect, mo):
    mo.output.append(mo.md(r"""####False Positives"""))
    fp = incorrect[incorrect['annotation'] == 'O']
    return (fp,)


@app.cell
def _(fp):
    fp
    return


@app.cell
def _(fp, plt):
    #| export
    _hist = fp['prediction'].value_counts()
    _fig, _ax = plt.subplots()
    _ax.bar(_hist.index, _hist.values)
    _ax.set_xlabel('label')
    _ax.set_ylabel('frequency')
    _ax.set_title('Frequency of labels among False Positives')
    _ax.tick_params(axis='x', rotation=90)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(incorrect, mo):
    mo.output.append(mo.md(r"""####False Negatives"""))
    fn = incorrect[incorrect['prediction'] == 'O']
    return (fn,)


@app.cell
def _(fn):
    fn
    return


@app.cell
def _(fn, plt):
    _hist = fn['annotation'].value_counts()
    _fig, _ax = plt.subplots()
    _ax.bar(_hist.index, _hist.values)
    _ax.set_xlabel('label')
    _ax.set_ylabel('frequency')
    _ax.set_title('Frequency of labels among False Negatives')
    _ax.tick_params(axis='x', rotation=90)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(incorrect, mo):
    mo.output.append(mo.md(r"""####Mislabels"""))
    mislabels = incorrect[(incorrect['annotation'] != 'O') & (incorrect['prediction'] != 'O')]
    return (mislabels,)


@app.cell
def _(mislabels):
    mislabels
    return


@app.cell
def _(mislabels, plt):
    _hist = mislabels['annotation'].value_counts()
    _fig, _ax = plt.subplots()
    _ax.bar(_hist.index, _hist.values)
    _ax.set_xlabel('label')
    _ax.set_ylabel('frequency')
    _ax.set_title('Frequency of annotation labels among mislabeled tokens')
    _ax.tick_params(axis='x', rotation=90)
    plt.close(_fig)
    _fig
    return


@app.cell
def _(mislabels, plt):
    _hist = mislabels['prediction'].value_counts()
    _fig, _ax = plt.subplots()
    _ax.bar(_hist.index, _hist.values)
    _ax.set_xlabel('label')
    _ax.set_ylabel('frequency')
    _ax.set_title('Frequency of prediciton labels among mislabled tokens')
    _ax.tick_params(axis='x', rotation=90)
    plt.close(_fig)
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
