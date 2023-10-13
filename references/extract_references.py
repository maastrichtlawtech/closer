import os

from legal_openai.openai_tasks import OpenaiTask


class ExtractReferences:
    def __init__(self):
        self.prompt_path = os.environ.get("PROMPT_PATH")

    def openai_references(self, prompt=None, path=None, api_key=None, article=None):
        """openai_references.

        Parameters
        ----------
        prompt :
            prompt
        """
        if prompt is None:
            with open(self.prompt_path + 'references.txt', "r") as f:
                prompt = f.read()
        return OpenaiTask(path=path, api_key=api_key).execute_task(article=article,
                                                                   prompt=prompt)
