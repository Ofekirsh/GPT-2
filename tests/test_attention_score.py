import torch
import gpt2.model.attention as attention


def test_attention_scores():
    # Test case 1: Simple example with identity vectors
    a = torch.tensor([[[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]]])  # Shape (1, 3, 3)

    b = torch.tensor([[[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 1.0]]])  # Shape (1, 3, 3)

    # Manually calculated expected output
    expected_output_1 = torch.tensor([[[0.5774, 0.0000, 0.0000],
                                       [0.0000, 0.5774, 0.0000],
                                       [0.0000, 0.0000, 0.5774]]], dtype=torch.float32)  # Shape (1, 3, 3)

    # Compute the attention scores using the function from the attention module
    A1 = attention.attention_scores(a, b)

    # Assert that the computed output is close to the expected output
    assert torch.allclose(A1, expected_output_1, atol=1e-4)


if __name__ == "__main__":
    test_attention_scores()
