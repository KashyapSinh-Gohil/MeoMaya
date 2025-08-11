import pprint

"""
full_nlp_workflow_demo.py

This script demonstrates the main features of the MeoMaya NLP framework step-by-step.
Each section is clearly labeled and explained for easy understanding.
"""
from meomaya.core.normalizer import Normalizer
from meomaya.core.parser import Parser
from meomaya.core.tagger import Tagger
from meomaya.core.tokenizer import Tokenizer
from meomaya.ml.classifier import Classifier
from meomaya.ml.vectorizer import Vectorizer
from meomaya.text.pipeline import TextPipeline

# 1. Load a text corpus
data = [
    "I love this product! It's absolutely wonderful.",
    "Terrible experience. Will not buy again.",
    "Average quality, nothing special.",
    "Fast shipping and great customer service.",
    "The item broke after one use.",
]
corpus = data

# 2. Normalize the text
normalizer = Normalizer()
normalized_corpus = [normalizer.normalize(text) for text in corpus]

# 3. Tokenize the text
tokenizer = Tokenizer()
tokenized_corpus = [tokenizer.tokenize(text) for text in normalized_corpus]

# 4. Tag tokens (POS tagging)
tagger = Tagger()
tagged_corpus = [tagger.tag(tokens) for tokens in tokenized_corpus]

# 5. Parse sentences
parser = Parser()
parsed_corpus = [parser.parse(tagged) for tagged in tagged_corpus]

# 6. Vectorize the text
vectorizer = Vectorizer()
vectors = vectorizer.fit_transform(normalized_corpus)

# 7. Classify the text (e.g., sentiment)
classifier = Classifier()
classifier.train(
    vectors, ["positive", "negative", "negative", "positive", "negative"]
)  # Dummy labels
predictions = classifier.classify(vectors)

# 8. Run sentiment analysis (using classifier output)
sentiments = predictions

# 9. Build a custom pipeline
pipeline = TextPipeline()
pipeline_results = [pipeline.process(text) for text in corpus]


# 10. CLI automation for batch processing (simulated)
def batch_process(texts):
    return [pipeline.process(text) for text in texts]


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(indent=2, width=100, compact=False)
    print("\n==============================")
    print(" MeoMaya NLP Workflow Demo")
    print("==============================\n")

    print("Step 1: Original Corpus (Raw Text)")
    pp.pprint(corpus)

    print("\nStep 2: Normalized Text (Cleaned)")
    pp.pprint(normalized_corpus)

    print("\nStep 3: Tokenized Text (Split into Words)")
    pp.pprint(tokenized_corpus)

    print("\nStep 4: Tagged Tokens (Part-of-Speech)")
    pp.pprint(tagged_corpus)

    print("\nStep 5: Parsed Sentences (Syntax Trees)")
    pp.pprint(parsed_corpus)

    print("\nStep 6: Vectors (Numerical Features)")
    pp.pprint(vectors)

    print("\nStep 7: Predictions (Sentiment Labels)")
    pp.pprint(predictions)

    print("\nStep 8: Sentiments (Same as Predictions)")
    pp.pprint(sentiments)

    print("\nStep 9: Pipeline Results (All Steps Combined)")
    pp.pprint(pipeline_results)

    print("\nStep 10: Batch Processed Results (Multiple Inputs)")
    batch_results = batch_process(["Great value for money!", "Worst purchase ever."])
    pp.pprint(batch_results)

    print("\n==============================")
    print(" End of Demo")
    print("==============================\n")
