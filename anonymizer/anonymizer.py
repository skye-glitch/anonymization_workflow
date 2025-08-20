# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "itables==2.4.4",
#     "marimo",
#     "pip==25.2",
#     "presidio-analyzer==2.2.359",
#     "requests==2.32.5",
#     "spacy==3.8.7",
#     "spacy-curated-transformers==0.3.1",
#     "spacy-transformers==1.3.9",
#     "presidio-anonymizer==2.2.359",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import sys
    import os
    from pathlib import Path

    from presidio_anonymizer import AnonymizerEngine
    return AnonymizerEngine, Path, os, sys


@app.cell
def _(mo):
    mo.md(r"""# Anonymizer""")
    return


@app.cell
def _():
    anonymized_entities =  [
        'EMAIL_ADDRESS',
        'PHONE_NUMBER',
        'US_SSN',
        'NRP',
        'PERSON',
        'LOCATION', 'ADDRESS',
        'USERNAME',
        'CLIENT_COMPANY_NAME',
        'PROJECT_NUMBER',
        'PASSWORD_CODE',
        'SECRET_URL_IP',
        'ID_NUMBER'
        'PORT_NUMBER'
    ]
    return (anonymized_entities,)


@app.cell
def _(Path, mo, sys):
    mo.output.append(mo.md(r"## Getting Analyzer"))
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    sys.path.append(f'{NOTEBOOK_DIR}/../analyzer')
    import analyzer
    analyzer_state = analyzer.app.run()
    myAnalyzer = analyzer_state[1]['analyzer']
    return NOTEBOOK_DIR, myAnalyzer


@app.cell
def _(AnonymizerEngine):
    engine = AnonymizerEngine()
    return (engine,)


@app.cell
def _(mo):
    test_text = mo.ui.text_area(value='Test text', label='Text to anonymize: ')
    mo.vstack([
        mo.md('## Debugging'),
        test_text
    ])
    return (test_text,)


@app.cell
def _(anonymized_entities, engine, mo, myAnalyzer, test_text):
    anonymous_test_text = engine.anonymize(test_text.value, myAnalyzer.analyze(test_text.value, language='en', entities=anonymized_entities)).text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
    #anonymous_text
    mo.output.append(mo.md(f"""
    Anonymous Text:
    <div style="border:1px solid black; padding:8px; border-radius:4px; display:inline-block; width:fit-content;">
    {anonymous_test_text}
    </div>
    """))
    return


@app.cell
def _(NOTEBOOK_DIR, mo):
    input_path = mo.ui.text(value=f'{NOTEBOOK_DIR}/test.txt', label='Path to file to anonymize: ')
    input_path
    return (input_path,)


@app.function
def get_output_path(inpath):
    inpaths = inpath.split('.')
    inpaths.insert(-1, 'anonymized')
    outpath = '.'.join(inpaths)
    return outpath


@app.cell
def _(anonymized_entities, engine, input_path, mo, myAnalyzer, os):
    input_text = ''
    if os.path.exists(input_path.value):
        mo.output.append(mo.md(f'<span style="color: green">{input_path.value} found!</span><br>Running anonymizer.'))
        with open(input_path.value, 'r') as f:
            input_text = f.read()
        anonymous_text = engine.anonymize(input_text, myAnalyzer.analyze(input_text, language='en', entities=anonymized_entities)).text
        outpath = get_output_path(input_path.value)
        mo.output.append(mo.md(f'Writing anonymized text to {outpath}'))
        with open(outpath, 'w') as f:
            f.write(anonymous_text)
    else:
        mo.output.append(mo.md(f'<span style="color: red">Error: {input_path.value} not found</span>'))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
