# Anonymization Workflow

## Introduction
This repository contains [marimo](https://marimo.io) scripts to anonymize help desk tickets.
This workflow uses the [Presidio framework](https://microsoft.github.io/presidio/) from Microsoft to build:

1. an analyzer to identify PII entities in the text of help desk tickets
2. an evaluator that takes annotated tickets and the analyzer to provide the performance metrics on the analyzer
3. a pipeline to train a custom [spacy](https://spacy.io) model
4. an anonymizer that uses the analyzer to anonymize PII data from help desk tickets

## Usage

This repository contains subdirectors for each of the components of the workflow.
A marimo notebook is provide for each component.

To start, ensure that you have a package manager such as uv, pip, or conda and a python version 3.9 or greater.
The marimo package can be easily installed with uv, pip, or conda.
Follow the instructions [here](https://docs.marimo.io/getting_started/installation/) if you have any issues.


We recommend running your notebooks with the command
```
marimo edit --sandbox </path/to/marimo_notebook.py>
```
if you plan to `edit` a file and
```
marimo run --sandbox </path/to/marimo_notebook.py>
```
if you just to `run` it as a script.

To customize the framework for your center, begin by `edit`ing the `analyzer/analyzer.py` notebook.
There are custom recognizers for different PII entities including:

- USERNAME
- CLIENT_COMPANY_NAME
- PROJECT_NUMBER
- PASSWORD_CODE
- SECRET_URL_IP
- PORT_NUMBER
- PUBLIC_URL_IP
- PUBLIC_ADDRESS

as well as the following entities predefined by presidio:

- EMAIL_ADDRESS
- PHONE_NUMBER
- US_SSN
- NRP
- PERSON
- LOCATION

To aid in development, the notebook includes code to debug your analyzer and test on sample text.

Once you are finished modifying the analyzer, you can `run` the `evaluator/evaluation.py` script to understand the analyzer.
Postprocessing charts, dataframes, and metrics are available to help in understanding the results of the evaluation.
Note that you will need a json file or spacy doc containing annotated tickets for your center in order to run the evaluation.

If you determine that a spacy model trained on tickets from your center would improve the accuracy of your analyzer,
`run` the `training/training.py` notebook to train a custom spacy model.
Open the `analyzer/analyzer.py` and `evaluator/evaluation.py` in a text editor and add the path to the wheel to the list of dependencies, this should install the package running the notebooks in sandbox mode. An example model whl is provided in the comments of the notebooks as an example.
Then edit the `model_name` in the `NlpEngineProvider` configuration in `analyzer/analyzer.py` to use this new model and re-`run` the evaluator to see how it performs.

Finally, once you are happy with your `analyzer`, you can `run` the `anonymizer/anonymizer.py` notebook to begin anonymizing your text files.

## Missing files
Some files are not included in this distribution as they contain to PII:

- A json file or spacy Doc file generated from annotated tickets used for evaluation are not included.
- CSV files provided in `./analyzer/databases` contain dummy data for usernames, commercial clients, and project codes. They will need to updated with actual data from your center to accurately identify the respective entities.

## Notes on Annotation

The analyzer and annotation scripts in this repository are based on the analysis and evaluation of OSC helpdesk tickets.

To evaluate helpdesk tickets from other sources, a sampling of tickets will need to be annotated and evaluated.

### Annotation tools

For this work, we used the [ner-annotator](https://github.com/tecoholic/ner-annotator), which is no longer actively supported,
but is still available and can be used for annotating of tickets.
The spacy training pipeline assumes that the annotations are in
`ner-annotator`'s output format but other annotation tools may be used as long as they can be converted to spacy Doc format.

### annotation directory

The annotation directory provides a notebook, `convert.py` that can be used to take a set of annotated tickets, split them into training, dev, and sets and then convert them into spacy Doc files.
We provide a text file containing a few sample tickets as well as a sample JSON file containing annotations in the expected format.
The entire set of annotated tickets is also converted into a spacy Doc file if the annotated tickets will only be used for evaluation and not for the training of a new model.