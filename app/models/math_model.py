from pix2text import Pix2Text
from PIL import Image

class MathModel:
    def __init__(self):
        # We initialize Pix2Text specifically for LaTeX extraction
        self.engine = Pix2Text(languages=('en',), analyzer_config=dict(model_name='mfd'))

    def extract_latex(self, image: Image.Image):
        """Extracts LaTeX from a math region."""
        # recognize_formula returns the LaTeX string
        result = self.engine.recognize_formula(image)
        return result
