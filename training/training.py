# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "pip==25.2",
#     "polars==1.32.3",
#     "spacy==3.8.7",
# ]
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    from glob import glob
    from pathlib import Path
    import json
    import polars as pl
    import spacy
    from os.path import exists
    from spacy.cli.train import train
    from spacy.cli.package import package
    from spacy.cli.download import download
    return Path, download, exists, mo, package, train


@app.cell
def _(mo):
    mo.md(r"""# Spacy Training""")
    return


@app.cell
def _(Path):
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    return (NOTEBOOK_DIR,)


@app.cell
def _(download):
    download('en_core_web_lg')
    return


@app.cell
def _(NOTEBOOK_DIR, mo, train):
    mo.output.append(mo.md(f"""
    Training a spacy model using:<br>
    - config file: {NOTEBOOK_DIR}/config.cfg<br>
    - output path: {NOTEBOOK_DIR}/spacy_model<br>
    - training set: {NOTEBOOK_DIR}/../annotation/training.spacy<br>
    - dev set: {NOTEBOOK_DIR}/../annotation/dev.spacy<br>
    """))
    train(
        config_path=f'{NOTEBOOK_DIR}/config.cfg',
        output_path=f'{NOTEBOOK_DIR}/spacy_model',
        overrides={'paths.train': f'{NOTEBOOK_DIR}/../annotation/training.spacy', 'paths.dev': f'{NOTEBOOK_DIR}/../annotation/dev.spacy'}
    )
    mo.output.append(mo.md(f'Training completed. Saved output to {NOTEBOOK_DIR}/spacy_model'))
    return


@app.cell
def _(NOTEBOOK_DIR, mo):
    package_dir = NOTEBOOK_DIR / 'packages'
    package_dir.mkdir(parents=True, exist_ok=True)
    packagename_ui = mo.ui.text(value='helpdesktickets', label='Spacy Package Name: ')
    packagename_ui
    return package_dir, packagename_ui


@app.cell
def _(mo):
    version_ui = mo.ui.text(value='0.1', label='Spacy Package version: ')
    version_ui
    return (version_ui,)


@app.cell
def _(packagename_ui, version_ui):
    version = version_ui.value
    packagename = packagename_ui.value
    return packagename, version


@app.cell
def _(NOTEBOOK_DIR, exists, mo, package, package_dir, packagename, version):

    try:
        if exists(NOTEBOOK_DIR / 'packages' / f'en_{packagename}-{version}'):
            mo.output.append(mo.md(f'Package `en_{packagename}-{version}` already exists. Delete the previous package or update version.'))
        else:
            mo.output.append(mo.md('Packaging the spacy model'))
            package(
                NOTEBOOK_DIR / 'spacy_model' / 'model-best',
                package_dir,
                create_wheel=True,
                name=packagename,
                version=version
            )
            mo.output.clear()
            mo.output.append(mo.md(f'Packaged spacy model to {NOTEBOOK_DIR}/{packagename}'))
            mo.output.append(mo.md(f"""
            ## Usage Instructions
            To use this model in your analyzer, run pip install `en_{packagename}-{version}-py3-non-any.whl` and change the model_name in the `NlpEngineProvider` configuration to `en_{packagename}`.
            """))

    except SystemExit as e:
        mo.output.clear()
        mo.output.append(mo.md(f'Packaging failed. Make sure you are not overwriting an existing package.'))
    return


if __name__ == "__main__":
    app.run()
