import unittest

from sentiment import run_sentiment_analysis

class TestSentimentAnalysis(unittest.TestCase):

    def test_positive_sentiment(self):
        input_text = "I love this product! It's amazing."
        expected_output = "joy"
        result = run_sentiment_analysis(input_text, model_name = "gpt-4o-mini", num_iterations = 5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_negative_sentiment(self):
        input_text = "I hate this product. It's terrible."
        expected_output = "anger"
        result = run_sentiment_analysis(input_text, model_name = "gpt-4o-mini", num_iterations = 5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_neutral_sentiment(self):
        input_text = "This product is okay."
        expected_output = "indifference"
        result = run_sentiment_analysis(input_text, model_name = "gpt-4o-mini", num_iterations = 5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    # testing edge case for empty input
    def test_empty_input(self):
        input_text = ""
        expected_output = "empty text"  # Replace with expected output for empty input
        result = run_sentiment_analysis(input_text, model_name = "gpt-4o-mini", num_iterations = 5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_mixed_sentiment(self):
        input_text = "I love the design, but the performance is terrible."
        expected_output = "ambivalence"  # Replace with expected output
        result = run_sentiment_analysis(input_text, model_name = "gpt-4o-mini", num_iterations = 5)['most_frequent_response']
        self.assertEqual(result, expected_output)
        
    def test_contentment(self):
        input_text = "I'm satisfied with how things are going."
        expected_output = "contentment"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_excitement(self):
        input_text = "I can't wait for the concert tonight!"
        expected_output = "excitement"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_gratitude(self):
        input_text = "Thank you so much for your help, I really appreciate it."
        expected_output = "gratitude"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_sadness(self):
        input_text = "I feel so down today, nothing seems to go right."
        expected_output = "sadness"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_fear(self):
        input_text = "I'm terrified of what might happen next."
        expected_output = "fear"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_disgust(self):
        input_text = "That meal was absolutely revolting."
        expected_output = "disgust"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_nostalgia(self):
        input_text = "Looking at these old photos makes me miss the good old days."
        expected_output = "nostalgia"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_bittersweet(self):
        input_text = "I'm happy for my friend's success, but sad that they're moving away."
        expected_output = "bittersweet"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_anticipation(self):
        input_text = "I'm eagerly looking forward to the results of the experiment."
        expected_output = "anticipation"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_surprise(self):
        input_text = "I never expected to win the lottery!"
        expected_output = "surprise"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_empathy(self):
        input_text = "I understand how you feel, it must be really tough for you."
        expected_output = "empathy"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_pride(self):
        input_text = "I'm really proud of my accomplishments this year."
        expected_output = "pride"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_shame(self):
        input_text = "I feel so embarrassed about what happened yesterday."
        expected_output = "shame"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_guilt(self):
        input_text = "I regret not helping my friend when they needed me."
        expected_output = "guilt"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_curiosity(self):
        input_text = "I wonder how they managed to build such an intricate machine."
        expected_output = "curiosity"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_confusion(self):
        input_text = "I'm having trouble understanding these complex instructions."
        expected_output = "confusion"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_certainty(self):
        input_text = "I'm absolutely sure this is the right decision."
        expected_output = "certainty"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_doubt(self):
        input_text = "I'm not sure if I made the right choice."
        expected_output = "doubt"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_mixed_emotions(self):
        input_text = "I'm excited about the new job, but also anxious about the challenges."
        expected_output = "ambivalence"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

    def test_complex_emotion(self):
        input_text = "As I watch my child graduate, I feel proud, happy, and a little sad that they're growing up so fast."
        expected_output = "bittersweet"
        result = run_sentiment_analysis(input_text, model_name="gpt-4o-mini", num_iterations=5)['most_frequent_response']
        self.assertEqual(result, expected_output)

if __name__ == '__main__':
    unittest.main()