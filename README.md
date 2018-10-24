# deep-texture-synthesis-cnn-keras

In this project I have implemented Textures Synthesis Using Convolutional Neural Networks paper by Gatys et.al. In this paper they have introduce a new model of natural textures generation based on the feature spaces
of convolutional neural networks optimised for object recognition. Within the model, textures are represented by the correlations between feature maps in several layers of
the network. They showed that across layers the texture representations increasingly
capture the statistical properties of natural images while making object information more and more explicit. Their model provides a new tool to generate stimuli
for neuroscience and might offer insights into the deep representations learned by
convolutional neural networks.
<br>

# Input Image
<img src='data/inputs/pebbles.jpg' />

<br>

# Output Image
The following output is generated after 500 iterations, you can control number of iterations by passing the value of iteration in the function buildTexture
<br>

  
  
# Reference
<ul>
<li>
Leon A. Gatys, Alexander S. Ecker: “Texture Synthesis Using Convolutional Neural Networks”, 2015; <a href='http://arxiv.org/abs/1505.07376'>arXiv:1505.07376</a>.
</li>
<li>
Implementation of Style Transfer by François Chollet in <a href='https://github.com/keras-team/keras/blob/master/examples/neural_style_transfer.py'>Neural Style Transfer</a>    
</li>
</ul>
