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
#     #    "en_helpdesktickets==0.1",
# ]
# [tool.uv.sources]
# # en_helpdesktickets = { path = "../training/packages/en_helpdesktickets-0.1/dist/en_helpdesktickets-0.1-py3-none-any.whl", editable = true }
# ///

import marimo

__generated_with = "0.14.17"
app = marimo.App()

with app.setup:
    import marimo as mo


@app.cell(hide_code=True)
def _():
    mo.md(r"""# Building the Analyzer""")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Setup""")
    return


@app.cell
def _():
    from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern, EntityRecognizer, RecognizerResult, AnalysisExplanation, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider, NlpArtifacts, TransformersNlpEngine, NerModelConfiguration, NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import SpacyRecognizer, PhoneRecognizer, EmailRecognizer, DateRecognizer, IpRecognizer
    from presidio_analyzer.context_aware_enhancers import LemmaContextAwareEnhancer
    import json
    import re
    import spacy
    import spacy_transformers
    import mimetypes
    import csv
    from spacy import displacy
    from pprint import pprint
    from itables import init_notebook_mode
    from typing import Dict, List, Optional, Tuple, Set
    from pathlib import Path
    init_notebook_mode(all_interactive=True)
    return (
        AnalysisExplanation,
        AnalyzerEngine,
        EmailRecognizer,
        EntityRecognizer,
        IpRecognizer,
        LemmaContextAwareEnhancer,
        List,
        NlpArtifacts,
        NlpEngineProvider,
        Optional,
        Path,
        Pattern,
        PatternRecognizer,
        PhoneRecognizer,
        RecognizerResult,
        SpacyRecognizer,
        csv,
        displacy,
        mimetypes,
        re,
    )


@app.cell
def _(Path):
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    return (NOTEBOOK_DIR,)


@app.cell
def _(NlpEngineProvider):
    # configure nlp engine with spacy model

    configuration = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": "en_core_web_trf"}]
        #"models": [{"lang_code": "en", "model_name": "en_helpdesktickets"}]
    }

    provider = NlpEngineProvider(nlp_configuration=configuration)
    nlp_engine = provider.create_engine()
    return (nlp_engine,)


@app.cell
def _():
    entities_to_keep =  [
        # default spacy entities
        'EMAIL_ADDRESS',
        'PHONE_NUMBER',
        'US_SSN',
        'NRP',
        'PERSON',
        'LOCATION',
        # custom spacy entities
        'USERNAME',
        'CLIENT_COMPANY_NAME',
        'PROJECT_NUMBER',
        'PASSWORD_CODE',
        'SECRET_URL_IP',
        'PORT_NUMBER'
    ]
    return (entities_to_keep,)


@app.cell
def _():
    whitelisted_emails = ['oschelp@osc.edu', 'no-reply@osc.edu', 'l2support@osc.edu', 'splunk@osc.edu', 'slurm@osc.edu', 'security@osc.edu', 'webmaster@osc.edu']
    return (whitelisted_emails,)


@app.cell
def _():
    whitelisted_phonenumbers = ["6142921800"]
    return (whitelisted_phonenumbers,)


@app.cell
def _(LemmaContextAwareEnhancer):
    # Set up the context aware enhancer
    context_enhancer = LemmaContextAwareEnhancer(context_prefix_count=10, 
                                                 context_suffix_count=10)
    return (context_enhancer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Build Recognizers""")
    return


@app.cell
def _(NOTEBOOK_DIR, csv):
    def get_usernames(pth=f'{NOTEBOOK_DIR}/databases/usernames.csv'):
        with open(pth, 'r', errors='ignore') as f:
            csvreader = csv.reader(f)
            header = next(csvreader)
            #usernames = [row[header.index('Username')] for row in csvreader]
            usernames = [row[0] for row in csvreader]
            usernames_lower = [user.lower() for user in usernames]
            return usernames, usernames_lower
    return (get_usernames,)


@app.cell
def _(NOTEBOOK_DIR, csv):
    def get_clients(pth=f'{NOTEBOOK_DIR}/databases/commercial_clients.csv'):
        with open(pth, 'r', errors='ignore') as f:
            csvreader = csv.reader(f)
            next(csvreader)
            return [comp for row in csvreader for comp in row]
    return (get_clients,)


@app.cell
def _(NOTEBOOK_DIR, csv):
    def get_projects(pth=f'{NOTEBOOK_DIR}/databases/projects.csv'):
        with open(pth, 'r', errors='ignore') as f:
            csvreader = csv.reader(f)
            header = next(csvreader)
            projects = [row[header.index('Project Code')] for row in csvreader]
            projects_lower = [p.lower() for p in projects]
            return projects, projects_lower
    return (get_projects,)


@app.cell
def _(
    EntityRecognizer,
    List,
    NlpArtifacts,
    RecognizerResult,
    get_usernames,
    re,
):
    class UsernamesRecognizer(EntityRecognizer):
        def __init__(self, deny_emails):
            super().__init__(self)
            self.supported_entities=['USERNAME']
            self.deny_emails = deny_emails
            self.usernames, self.usernames_lower = get_usernames()
            self.keywords = ['user', 'users', 'username', 'usernames', '/', 'finger'] + self.usernames


        def load(self) -> None:
            """No loading is required."""
            pass

        def analyze(
            self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
        ) -> List[RecognizerResult]:
            results = []
            #print(f'nlp artifacts passed to username recognizer: <<<{nlp_artifacts.tokens.text}>>>')
            # iterate over the spaCy tokens
            for idx, token in enumerate(nlp_artifacts.tokens):
                token_text = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', token.text)
                match = False
                if token_text in self.usernames:
                    match = True
                    conf = 0.0
                elif token_text.lower() in self.usernames_lower:
                    match = True
                    conf = 0.0

                if match:
                    if token.is_oov:
                        conf = conf + 0.86
                    elif nlp_artifacts.tokens[idx-1].text.lower() in self.keywords:
                        conf = conf + 0.5
                    elif [t.text for t in nlp_artifacts.tokens[min(idx+1, len(nlp_artifacts.tokens)-1):min(idx+5, len(nlp_artifacts.tokens)-1)]] == ['(', 'Additional', 'comments', ')']:
                        conf = conf + 0.9

                    result = RecognizerResult(
                        entity_type="USERNAME",
                        start=token.idx,
                        end=token.idx + len(token),
                        score=conf,
                    )
                    results.append(result)

                if any(c in ['@', '/'] for c in token_text) and not any(c in token_text for c in self.deny_emails):
                    for m in re.finditer(r'[^@/]+', token.text):
                        sub = m.group()
                        if sub in self.usernames:
                            result = RecognizerResult(
                                entity_type='USERNAME',
                                start=token.idx + m.start(),
                                end=token.idx + m.end(),
                                score = 0.7
                            )
                            results.append(result)

                # check for osu name.#s
                if re.search(r'\b[a-zA-Z]{2,}\.\d+\b', token_text):
                    result = RecognizerResult(
                        entity_type="USERNAME",
                        start=token.idx,
                        end=token.idx + len(token),
                        score=0.5,
                    )
                    results.append(result)

                # check for social media accounts
                if token_text.lower() in {'linkedin', 'twitter', 'bluesky'}:
                    for next_idx in range(idx+1, idx + 6):
                        next_token = nlp_artifacts.tokens[next_idx]
                        if next_token.text.startswith('@'):
                            result = RecognizerResult(
                                entity_type='USERNAME',
                                start = next_token.idx,
                                end = next_token.idx + len(next_token),
                                score=0.7
                            )
                            results.append(result)
                            break


            return results
    return (UsernamesRecognizer,)


@app.cell
def _(
    AnalysisExplanation,
    EntityRecognizer,
    List,
    NlpArtifacts,
    RecognizerResult,
    get_projects,
):
    class ProjectsRecognizer(EntityRecognizer):
        def __init__(self):
            super().__init__(self)
            self.projects, self.projects_lower = get_projects()
            self.supported_entities=["PROJECT_NUMBER"]

        def load(self) -> None:
            """No loading is required."""
            pass

        def analyze(
            self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
        ) -> List[RecognizerResult]:
            results = []

            # iterate over the spaCy tokens, and call `token.like_num`
            for token in nlp_artifacts.tokens:
                #if token.text in projects:
                if any(project.lower() in token.text for project in self.projects_lower) or any(project in token.text for project in self.projects):
                    result = RecognizerResult(
                        entity_type="PROJECT_NUMBER",
                        start=token.idx,
                        end=token.idx + len(token),
                        score=0.7,
                        analysis_explanation=AnalysisExplanation(recognizer=self.__class__.__name__,
                                                                 original_score=0.7,
                                                                 textual_explanation='Token contains a project ID from project database')
                    )
                    results.append(result)
            return results
    return (ProjectsRecognizer,)


@app.function
def sanitize_token(token: str) -> str:
    import re
    token = re.sub(r'[^a-zA-Z0-9]', '', token).lower()
    return token


@app.cell
def instiutions_recognizer(
    EntityRecognizer,
    List,
    NlpArtifacts,
    RecognizerResult,
    get_clients,
    re,
):
    class InstitutionsRecognizer(EntityRecognizer):
        expected_confidence_level = 0.9  # expected confidence level for this recognizer

        def __init__(self):
            super().__init__(self)
            self.companies = get_clients()
            self.supported_entities=["CLIENT_COMPANY_NAME"]

        def load(self) -> None:
            """No loading is required."""
            pass

        def analyze(
            self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
        ) -> List[RecognizerResult]:
            """
            Analyzes text to find tokens which represent institutions (single or multi-word names).
            """
            results = []

            # Split the tokens from insts into multi-token institutions
            for institution in self.companies:
                institution_tokens = re.split(r'\s+', institution)

                # Check if the tokens appear in sequence in the text
                for idx, token in enumerate(nlp_artifacts.tokens):
                    if token.text.lower() == institution_tokens[0].lower():
                        # Check if subsequent tokens match
                        match = True
                        score = self.expected_confidence_level
                        num_tokens = 1
                        for i, institution_token in enumerate(institution_tokens[1:], 1):
                            if idx + i >= len(nlp_artifacts.tokens) or sanitize_token(nlp_artifacts.tokens[idx + i].text) != sanitize_token(institution_token):
                                if sanitize_token(institution_token) in ['inc', 'llc', 'co', 'ltd']:
                                    score = score - 0.1
                                    break
                                else:
                                    match = False
                                    break

                            num_tokens = num_tokens + 1

                        # If a match is found, create a result
                        if match:
                            start = token.idx
                            end = nlp_artifacts.tokens[idx + num_tokens - 1].idx + len(nlp_artifacts.tokens[idx + num_tokens - 1].text)

                            result = RecognizerResult(
                                entity_type="CLIENT_COMPANY_NAME",
                                start=start,
                                end=end,
                                score=score,
                            )
                            results.append(result)

            return results
    return (InstitutionsRecognizer,)


@app.cell
def _(IpRecognizer, List, NlpArtifacts, RecognizerResult, re):
    class myIpRecognizer(IpRecognizer):
        def __init__(self):
            super().__init__()
            self.supported_entity = 'SECRET_URL_IP'
            self.supported_entities = [self.supported_entity]

        def analyze(
            self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
        ) -> List[RecognizerResult]:
            results = super().analyze(text, entities, nlp_artifacts)
            for r in results:
                r.entity_type = self.supported_entity
            filtered = []
            for r in results:
                if not re.match(r'.*::(\w+)?', text[r.start:r.end]):
                    filtered.append(r)
            return filtered

        def get_supported_entities(self):
            return [self.supported_entity]
    return (myIpRecognizer,)


@app.cell
def _(
    List,
    NlpArtifacts,
    Optional,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
    re,
):
    class myIpwithQuestionMarkRecognizer(PatternRecognizer):
        PATTERNS = [
            Pattern(
                name="IPv4",
                regex=r"\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:\?){0,1}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:\?){0,1}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:\?){0,1}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
                score=0.6,
            ),
        ]

        CONTEXT = ["ip", "ipv4"]

        def __init__(
            self,
            patterns: Optional[List[Pattern]] = None,
            context: Optional[List[str]] = None,
            supported_language: str = "en",
            supported_entity: str = "SECRET_URL_IP",
        ):
            patterns = patterns if patterns else self.PATTERNS
            context = context if context else self.CONTEXT

            super().__init__(
                supported_entity=supported_entity,
                patterns=patterns,
                context=context,
                supported_language=supported_language,
            )
            self.supported_entity = supported_entity

        def analyze(
            self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
        ) -> List[RecognizerResult]:
            results = super().analyze(text, entities, nlp_artifacts, regex_flags=0)
            for r in results:
                r.entity_type = self.supported_entity
            filtered = []
            for r in results:
                if not re.match(r'.*::(\w+)?', text[r.start:r.end]):
                    filtered.append(r)
            return filtered
    return (myIpwithQuestionMarkRecognizer,)


@app.cell
def _(EntityRecognizer, List, NlpArtifacts, RecognizerResult, re):
    class PasswordCodeRecognizer(EntityRecognizer):
        def __init__(self):
            super().__init__(self)
            self.keywords = ['access', 'code', 'passcode']
            self.supported_entities=['PASSWORD_CODE']

        def load(self) -> None:
            """No loading is required."""
            pass

        def analyze(
            self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
        ) -> List[RecognizerResult]:
            """
            Analyzes text to find tokens which represent password codes.
            """
            pattern = re.compile(r"\b(?:[A-Z2-9]{6}|[0-9]{6})\b")
            results = []

            for m in pattern.finditer(text):
                for idx, token in enumerate(nlp_artifacts.tokens):
                    if token.idx == m.start() and token.text == m.group():
                        if token.is_oov:
                            context = [t.text.lower() for t in nlp_artifacts.tokens[max(0, idx - 10):min(idx + 11,len(nlp_artifacts.tokens))]]
                            if any(t in self.keywords for t in context):
                                result = RecognizerResult(
                                    entity_type='PASSWORD_CODE',
                                    start = m.start(),
                                    end = m.end(),
                                    score=0.4
                                )
                                results.append(result)

            return results
    return (PasswordCodeRecognizer,)


@app.cell
def _(EntityRecognizer, List, NlpArtifacts, RecognizerResult, mimetypes, re):
    class PortNumberRecognizer(EntityRecognizer):
        def load(self) -> None:
            """No loading is required."""
            pass

        def is_time(self, text: str, index: int) -> bool:
            p = re.compile(r"(?:(\d{1,3})T)?(\d{1,2}):([0-5]?\d)(?::([0-5]?\d))?(?:\.(\d{1,3}))?")
            m = re.search(r'\s+', text[:index][::-1])
            if m:
                start_index = index - m.start() - 1
                substrs = re.split(r'\s+', text[start_index+1:])
                if substrs:
                    substr = substrs[0]
                else:
                    substr = ''
            else:
                substr = ''
            if p.search(substr):
                return True
            return False

        def is_file_path(self, text: str, index: int) -> bool:
            m = re.search(r'\s+', text[:index][::-1])
            if m:
                start_index = index - m.start() - 1
                substrs = re.split(r'\s+', text[start_index+1:index])
                if substrs:
                    substr = substrs[0]
                else:
                    substr = ''
            else:
                substr = ''
            ext = substr.split('.')[-1]
            # check if ext is a known extension
            if mimetypes.guess_type('f.'+ext)[0]:
                return True
            return False


        def analyze(
            self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
        ) -> List[RecognizerResult]:
            results = []

            pattern = re.compile(r":\s*(\d{1,5})\b")
            for m in pattern.finditer(text):
                port = int(m.group(1))
                if port <= 65535 and port >= 0 and not self.is_time(text, m.start()) and not self.is_file_path(text, m.start()):
                    result = RecognizerResult(
                        entity_type = 'PORT_NUMBER',
                        start = m.start()+1,
                        end = m.end(),
                        score=0.25
                    )
                    results.append(result)

            tokens = nlp_artifacts.tokens
            for idx, token in enumerate(tokens):
                if token.text.lower() in ['on', 'port'] and idx + 1 < len(tokens):
                    next_token = tokens[idx+1].text
                    if next_token.isdigit():
                        next_token = int(next_token)
                        if next_token >= 0 and next_token <= 65535:
                            m = re.search(r'\b' +  re.escape(str(next_token)) + r'\b', text)
                            if m:
                                result = RecognizerResult(
                                    entity_type = 'PORT_NUMBER',
                                    start = m.start(),
                                    end = m.end(),
                                    score = 0.25)
                                results.append(result)

            return results
    return (PortNumberRecognizer,)


@app.cell
def _(List, NlpArtifacts, PhoneRecognizer, RecognizerResult, re):
    class myPhoneRecognizer(PhoneRecognizer):
        def __init__(self, supported_entities, denylist=None):
            super().__init__(supported_entities)
            self.denylist = denylist or []

        def is_time(self, text: str, index: int) -> bool:
            p = re.compile(r"(?:(\d{1,3})T)?(\d{1,2}):([0-5]?\d)(?::([0-5]?\d))?(?:\.(\d{1,3}))?")
            m = re.search(r'\s+', text[:index][::-1])
            if m:
                start_index = index - m.start() - 1
                substrs = re.split(r'\s+', text[start_index+1:])
                if substrs:
                    substr = substrs[0]
                else:
                    substr = ''
            else:
                substr = ''
            if p.search(substr):
                return True
            return False

        def phone_match(self, substr: str, numbers: list):
            normalized_substr = re.sub(r'\D', '', substr)
            return any(number in normalized_substr for number in numbers)

        def analyze(
            self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
        ) -> List[RecognizerResult]:
            results = super().analyze(text, entities, nlp_artifacts)

            # Filter out any results that match denylisted phone numbers
            filtered = [
                r for r in results
                #if text[r.start:r.end] not in self.denylist
                if not self.phone_match(text[r.start:r.end], self.denylist)
                and not self.is_time(text, r.start)
            ]
            return filtered
    return (myPhoneRecognizer,)


@app.cell
def _(List, NlpArtifacts, Pattern, PatternRecognizer, RecognizerResult):
    class myEmailWithQuestionMarks(PatternRecognizer):
        def __init__(self, denylist=None):
            pattern = Pattern(
                name="email_with_question_marks",
                regex=r"\b[A-Za-z0-9._%+-]+(?:\?[A-Za-z0-9._%+-]+)*@(?:\?[A-Za-z0-9-]+)+(?:\.(?:\?[A-Za-z0-9-]+)+)+\b",
                score=0.6
            )
            super().__init__(
                supported_entity="EMAIL_ADDRESS",
                name="EmailWithQuestionMarksRecognizer",
                patterns=[pattern],
                context=["email", "contact", "mail"]
            )
            self.denylist = denylist or []
        def analyze(
            self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
        ) -> List[RecognizerResult]:
            results = super().analyze(text, entities, nlp_artifacts, regex_flags = 0)

            # Filter out any results that match denylisted emails
            filtered = [
                r for r in results
                if text[r.start:r.end].lower() not in self.denylist
            ]
            return filtered
    return (myEmailWithQuestionMarks,)


@app.cell
def _(EmailRecognizer, List, NlpArtifacts, RecognizerResult):
    class myEmailRecognizer(EmailRecognizer):
        def __init__(self, denylist=None):
            super().__init__()
            self.denylist = denylist or []

        def analyze(
            self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
        ) -> List[RecognizerResult]:
            results = super().analyze(text, entities, nlp_artifacts, regex_flags = 0)

            # Filter out any results that match denylisted emails
            filtered = [
                r for r in results
                if text[r.start:r.end].lower() not in self.denylist
            ]
            return filtered
    return (myEmailRecognizer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""## Add Recognizers to Analyzer""")
    return


@app.cell(hide_code=True)
def _(
    AnalyzerEngine,
    InstitutionsRecognizer,
    PasswordCodeRecognizer,
    PortNumberRecognizer,
    ProjectsRecognizer,
    SpacyRecognizer,
    UsernamesRecognizer,
    context_enhancer,
    entities_to_keep,
    myEmailRecognizer,
    myEmailWithQuestionMarks,
    myIpRecognizer,
    myIpwithQuestionMarkRecognizer,
    myPhoneRecognizer,
    nlp_engine,
    whitelisted_emails,
    whitelisted_phonenumbers,
):
    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        context_aware_enhancer=context_enhancer,  # , log_decision_process=True)
    )


    analyzer.registry.add_recognizer(UsernamesRecognizer(deny_emails=whitelisted_emails))
    analyzer.registry.add_recognizer(
        InstitutionsRecognizer()
    )
    analyzer.registry.add_recognizer(
        ProjectsRecognizer()
    )
    analyzer.registry.add_recognizer(PasswordCodeRecognizer())
    analyzer.registry.add_recognizer(
        PortNumberRecognizer(supported_entities=["PORT_NUMBER"])
    )
    analyzer.registry.add_recognizer(
        myPhoneRecognizer(
            supported_entities=["PHONE_NUMBER"], denylist=whitelisted_phonenumbers
        )
    )
    analyzer.registry.remove_recognizer("PhoneRecognizer")
    analyzer.registry.add_recognizer(
        myEmailRecognizer(denylist=whitelisted_emails)
    )
    analyzer.registry.add_recognizer(
        myEmailWithQuestionMarks(denylist=whitelisted_emails)
    )
    analyzer.registry.add_recognizer(myIpRecognizer())
    analyzer.registry.add_recognizer(myIpwithQuestionMarkRecognizer())
    analyzer.registry.remove_recognizer("EmailRecognizer")
    for recognizer in analyzer.registry.recognizers:
        if not any(
            entity in entities_to_keep for entity in recognizer.supported_entities
        ):
            analyzer.registry.remove_recognizer(recognizer.name)
            print(
                f"- removing {recognizer.name} which supports {recognizer.supported_entities}"
            )
        elif isinstance(recognizer, SpacyRecognizer):
            analyzer.registry.remove_recognizer(recognizer.name)
            print(
                f"- removing {recognizer.name} which supports {recognizer.supported_entities}"
            )
        else:
            print(
                f"+ keeping {recognizer} which supports {recognizer.supported_entities}"
            )

    # Add back restricted spacy recognizer
    analyzer.registry.add_recognizer(
        SpacyRecognizer(supported_entities=["PERSON", "ID", "NRP", "ADDRESS", "PUBLIC_URL_IP", "SECRET_URL_IP", "CLIENT_COMPANY_NAME", "CLIENT_NAME", "PUBLIC_ADDRESS", "LOCATION", "LOC"])
    )
    print(analyzer.registry.get_supported_entities())
    return (analyzer,)


@app.cell
def _(
    AnalyzerEngine,
    EmailRecognizer,
    EntityRecognizer,
    List,
    NlpArtifacts,
    RecognizerResult,
    mimetypes,
    re,
):
    from presidio_analyzer.predefined_recognizers.url_recognizer import UrlRecognizer
    from urllib.parse import urlparse
    from requests import get

    class myURLRecognizer(EntityRecognizer):

        def __init__(self, supported_entities, analyzer_engine: AnalyzerEngine, internal_pii_list):
            super().__init__(supported_entities=supported_entities)
            self.url_recognizer = UrlRecognizer()
            self.analyzer_engine = analyzer_engine
            self.entities_to_keep = internal_pii_list
            self.entities_to_keep = [x for x in self.entities_to_keep if x not in ['PORT_NUMBER', 'SECRET_URL_IP', 'PUBLIC_URL_IP']]
            self.valid_extensions = [ext.lower() for ext in get('https://data.iana.org/TLD/tlds-alpha-by-domain.txt').text.split('\n')[1:]]
            self.file_exts = mimetypes.types_map.keys()


        def load(self) -> None:
            """No loading is required."""
            pass

        def is_part_of_email(self, text: str, index: int) -> bool:
            #start_index = text.rfind(' ', 0, index)
            m = re.search(r'\s+', text[:index][::-1])
            if m:
                start_index = index - m.start() - 1
                substrs = re.split(r'\s+', text[start_index+1:])
                if substrs:
                    substr = substrs[0]
                else:
                    substr = ''
            else:
                substr = ''
            if EmailRecognizer().analyze(substr, ['EMAIL_ADDRESS']) != []:
                return True
            return False     

        def valid_domain(self, url: str) -> bool:
            #parsed = urlparse(url if url.startswith("http") else "http://" + url)
            parsed = urlparse(url if "http" in url else "http://" + url)
            ext = parsed.netloc.split('.')[-1]
            return ext.lower() in self.valid_extensions and (f'.{ext.lower()}' not in self.file_exts if parsed.path == '' else True)

        def handle_osc_urls(self, text: str) -> str:
            if re.search(r'https://www\.osc\.edu/node/\d+/submission/\d+', text):
                return 'SECRET_URL_IP'
            return 'PUBLIC_URL_IP'

        def analyze(
            self, text: str, entities: List[str], nlp_artifacts: NlpArtifacts
        ) -> List[RecognizerResult]:
            """
            Analyzes text to find tokens which represent URLs
            """
            url_results = self.url_recognizer.analyze(text, ['URL'], nlp_artifacts=nlp_artifacts)
            #print(f'url results: {url_results}')
            results = []
            for url in url_results:
                if self.is_part_of_email(text, url.start):
                    continue
                if not self.valid_domain(text[url.start:url.end]):
                    continue
                url_text = text[url.start:url.end]
                if url.end < len(text):
                    if text[url.end].isalnum():
                        continue
                inner_results = self.analyzer_engine.analyze(text=text[url.start:url.end], language='en', entities=self.entities_to_keep)
                score = url.score
                if inner_results == []:
                    entity_type = 'PUBLIC_URL_IP'
                    if 'osc.edu' in text[url.start:url.end]:
                        entity_type = self.handle_osc_urls(text[url.start:url.end])
                    elif any(keyword in url_text for keyword in ['personal']):
                        entity_type = 'SECRET_URL_IP'
                else:
                    score = 0.9
                    entity_type = 'SECRET_URL_IP'
                myresult = RecognizerResult(
                    entity_type = entity_type,
                    start = url.start,
                    end = url.end,
                    score = score)
                results.append(myresult)
            return results
    return (myURLRecognizer,)


@app.cell
def _(analyzer, entities_to_keep, myURLRecognizer):
    analyzer.registry.add_recognizer(myURLRecognizer(supported_entities=['SECRET_URL_IP', 'PUBLIC_URL_IP'], analyzer_engine=analyzer, internal_pii_list=entities_to_keep))
    return


@app.cell
def _():
    mo.md(r"""## Analyzer Testing""")
    return


@app.cell
def _():
    test_text = mo.ui.text_area(value="Enter text here to test your analyzer", label='Test text: ')
    mo.md(f'{test_text}')
    return (test_text,)


@app.cell
def _(analyzer, test_text):
    results = analyzer.analyze(test_text.value, language='en')
    return (results,)


@app.cell
def _(displacy, results, test_text):
    # Display analyzed text
    ents = [{"start": rt.start, "end": rt.end, "label": rt.entity_type} for rt in results]
    mo.md(displacy.render({'ents': ents, 'text': test_text.value, 'title': None}, style='ent', manual=True, jupyter=False))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
