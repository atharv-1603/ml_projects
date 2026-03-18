import unittest
import os
from train import train_model

class TestModel(unittest.TestCase):

    def test_training(self):
        acc = train_model()
        self.assertGreater(acc, 0)

    def test_accuracy(self):
        acc = train_model()
        self.assertGreater(acc, 0.85)

    def test_model_file(self):
        train_model()
        self.assertTrue(os.path.exists("models/model.pkl"))

if __name__ == "__main__":
    unittest.main()
