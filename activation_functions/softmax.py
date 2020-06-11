import numpy as np 

class Softmax:
    def predict(self, inputs):
        exp = np.exp(inputs)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def loss(self, inputs, outputs):
        num_examples = inputs.shape[0]
        probs = self.predict(inputs)
        corect_logprobs = -np.log(probs[range(num_examples), outputs])
        data_loss = np.sum(corect_logprobs)
        return 1./num_examples * data_loss

    def diff(self, inputs, outputs):
        num_examples = inputs.shape[0]
        probs = self.predict(inputs)
        probs[range(num_examples), outputs] -= 1
        return probs