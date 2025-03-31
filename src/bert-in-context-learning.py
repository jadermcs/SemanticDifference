import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import List, Dict, Any


class BERTInContextLearningTest:
    def __init__(self, model_name: str = 'answerdotai/ModernBERT-large'):
        """
        Initialize BERT model and tokenizer for in-context learning testing

        :param model_name: Pretrained BERT model to use
        """
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)

        # Set model to evaluation mode
        self.model.eval()

    def prepare_context(self, examples: List[Dict[str, str]], query: str) -> torch.Tensor:
        """
        Prepare context with few-shot learning examples

        :param context: Main context to test
        :param examples: List of few-shot learning examples
        :return: Tokenized input tensor
        """
        # Construct prompt with examples
        prompt = ""
        for example in examples:
            prompt += f"Question: {example['question']}\n"
            prompt += f"Answer: {example['answer']}\n\n"

        # Add main context
        full_text = prompt
        full_text += f"Question: {query}\nAnswer: {self.tokenizer.mask_token}."
        print(full_text)

        # Tokenize input
        inputs = self.tokenizer(
            full_text,
            return_tensors='pt',
        )

        return inputs

    def generate_response(self, examples: List[Dict[str, str]], query: str) -> str:
        """
        Generate response using in-context learning

        :param context: Main context
        :param examples: Few-shot learning examples
        :param query: Question to answer
        :return: Generated response
        """
        # Prepare input
        inputs = self.prepare_context(examples, query)

        # Generate hidden states
        with torch.no_grad():
            logits = self.model(**inputs).logits
        mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
        predicted_token_id = logits[0, mask_token_index].argmax(axis=-1)

        # Example of simple response generation mechanism
        # Note: This is a simplified approach and not a true generative method

        return self.tokenizer.decode(predicted_token_id)

    def evaluate_in_context_performance(
        self,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Evaluate in-context learning performance

        :param test_cases: List of test scenarios
        :return: Performance metrics
        """
        correct_predictions = 0
        total_tests = len(test_cases)

        for case in test_cases:
            examples = case['few_shot_examples']
            query = case['query']
            expected_answer = case['expected_answer']

            # Generate response
            generated_response = self.generate_response(examples, query)
            print(generated_response)

            # Simple exact match accuracy
            if generated_response.lower() == expected_answer.lower():
                correct_predictions += 1

        return {
            'accuracy': correct_predictions / total_tests,
            'correct_predictions': correct_predictions,
            'total_tests': total_tests
        }


# Example usage and test cases
def main():
    # Initialize BERT in-context learning tester
    bert_tester = BERTInContextLearningTest()

    # Define test cases
    test_cases = [
        {
            'few_shot_examples': [
                {
                    'question': "Does the word 'head' has the same meaning in 'the head of the tower' and 'my head hurts'?",
                    'answer': "Different."
                },
                {
                    'question': "Does the word 'plane' has the same meaning in 'the plane flies' and 'I love planes and want to be a pilot'?",
                    'answer': "Identical."
                },
                {
                    'question': "Does the word 'heart' has the same meaning in 'the heart beats' and 'I need a heart surgery'?",
                    'answer': "Identical."
                },
            ],
            'query': "Does the word 'academy' has the same meaning in 'the French Academy' and 'the Music Academy'?",
            'expected_answer': "ip"
        },
        # Add more test cases here
    ]

    # Run performance evaluation
    performance = bert_tester.evaluate_in_context_performance(test_cases)

    # Print results
    print("In-Context Learning Performance:")
    print(f"Accuracy: {performance['accuracy'] * 100:.2f}%")
    print(f"Correct Predictions: {performance['correct_predictions']}")
    print(f"Total Tests: {performance['total_tests']}")


if __name__ == "__main__":
    main()
