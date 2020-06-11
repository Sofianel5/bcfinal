class ActivationFunction():

    def forward(self, inputs):
        """ The actual output of the function """
        return inputs 
        
    def backwards(self, inputs, differential):
        """ The derivative of the function times a differential (chain rule)"""
        return inputs