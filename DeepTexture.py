'''
    @author: Md Sarfarazul Haque
    This file contains the main class, of this project, 
    that synthesise a texture based on the provided texture.

    This Project is an implementation of following research paper:
    Citations{
        @online{
            1505.07376,
            Author = {Leon A. Gatys and Alexander S. Ecker and Matthias Bethge},
            Title = {Texture Synthesis Using Convolutional Neural Networks},
            Year = {2015},
            Eprint = {1505.07376},
            Eprinttype = {arXiv},
        }
    }

    NB: I have used fchollet's style transfer project as reference for this project.
    NB: Citations are in BibLaTeX format.

    As mentioned by authors replacing MaxPooling2D layers of VGG19 with AveragePooling2D layers result in 
    good results.

    So to fulfill this I have customized the original VGG19 model written by fchollet by replacing 
    MaxPooling2D layers with AveragePooling2D layers.
'''


''' Importing Packages '''
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from keras.preprocessing import image
import time
from scipy.misc import imsave
from keras.applications import vgg19

# This is a customized VGG19 network taken from fchollet implementation of VGG19
# from https://github.com/fchollet/deep-learning-models/blob/master/vgg19.py
from vgg19 import VGG19      
from keras import backend as K 
import tensorflow as tf



class DeepTexture(object):
    ''' This is the main class that is going to synthesise textures based on given one '''

    def __init__(self, tex_path, gen_prefix='result', base_img_path=None):
        ''' Method to initialize variables '''

        '''
            @tex_path: Path to the `Texture Image`, the image at this location will be used as reference for 
                        texture synthesis
            @gen_prefix: Prefix associated with the output image's names
            @base_img_path: This is the path to an image that can be used as base image to generate texture on.
                            It's default value being `None`. If not `None` it should point to an image that can be
                            used instead of random noise to synthesis texture on
        '''
        
        # Getting size of the input texture image.
        self.width, self.height = image.load_img(tex_path).size

        # Initializing loss value and gradient values as `None`
        self.loss_value = None
        self.grad_values = None
        self.channels = 3

        # To handle the case when base image is `None`
        # This generate a random noise matrix of size of our texture matrix.
        if base_img_path == None:
            x = np.random.rand(self.width, self.height, 3)

            # Converting [Width, Height, Channels] to [1, Width, Height, Channels]
            x = np.expand_dims(x, axis=0)

            # Preprocessing the noise image for inferencing through VGG19 model
            self.base_img = vgg19.preprocess_input(x.astype(np.float32))
        else:
            self.base_img = self.preprocess_image(base_img_path) # If base_img_path is not `None` then use that image as base image

        # Setting texture image path and prefix values
        self.tex_path = tex_path
        self.gen_prefix = gen_prefix
        # Setting the value of input_shape
        if K.image_data_format() == 'channels_last':
            self.input_shape = (1, self.height, self.width, self.channels)
        else:
            self.input_shape = (1, self.channels, self.height, self.width)



    def preprocess_image(self, img_path):
        '''
            This function makes an image ready to be inferentiable by preprocessing it according
            to VGG19 paper.

            @img_path: Path to an image to be preprocessed

            @return: Preprocessed image
        '''

        # Load the image using keras helper class `image`
        img = image.load_img(img_path, target_size=(self.width, self.height))
        # Converting image to array
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)

        # Applying preprocessing to the image
        img = vgg19.preprocess_input(img.astype(dtype=np.float32))
        return img


    
    def deprocess_image(self, x):
        '''
            This method deprocess the preprocessed image so that it can be saved to disk as normal images.

            @x: Image matrix

            @return: Converted image
        '''

        # Checking the data format supported by the backend
        if K.image_data_format() == 'channels_first':
            x = x.reshape((self.channels, self.height, self.width))
            x = x.transpose((1, 2, 0))
        else:
            x = x.reshape((self.height, self.width, self.channels))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    

    def gram_matrix(self, x):
        '''
            It is a function around which this application is built on. 
            This function calculates the gram matrix of the output of an intermediate layer.

            For more information about gram matrix do give a shot to above mentioned paper.

            @x: Feature map at some intermediate layer of VGG19 model

            @return: Calculated Gram Matrix
        '''

        # To check if input is a tensor or not
        # If not converting that x feature map to a tensor
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)

        # Getting the shape of the tensor
        shape = tf.shape(x)

        # Calculating F matrix by reshaping input tensor
        F = K.reshape(x, [shape[0], shape[1]*shape[2], shape[3]])

        # Calculating and preprocessing G, the Gram Matrix associated with a layer
        gram = tf.matmul(F, F, adjoint_a=True)
        gram /= 2*tf.to_float(shape[1]*shape[2])
        return gram



    def get_loss_per_layer(self, tex, gen):
        '''
            This function calculates the loss associated with a particular layer's output

            @tex: Layer's feature map coming from texture image
            @gen: Layer's feature map coming from synthesised image

            @return: Calculated loss assiciated with the current layer.
        '''

        # Get Gram Matrix of tex feature map
        Tex = self.gram_matrix(tex)
        # Get Gram Matrix of synthesised feature map
        Gen = self.gram_matrix(gen)
        return K.sum(K.square(tf.subtract(Tex, Gen)))


    def eval_loss_and_grads(self, x):
        '''
            This function calculates the total loss associated with synthesised with respect to the texture image.
            This function also calculates the total gradient of total loss with respect to the synthesised image. 

            @x: Current intermediate synthesised image

            @return loss_value, grad_values
                @loss_value: Total loss associated with the intermediate sysnthesised 
                            image with respect to texture image
                @grad_values: Total gradient of the loss function with respect to intermediate
                            synthesised image.
        '''

        # Checking and reshaping the 
        if K.image_data_format() == 'channels_first':
            x = x.reshape((1, 3, self.height, self.width))
        else:
            x = x.reshape((1, self.height, self.width, 3))

        # Getting loss_value and grad_values in the form of a list.
        outs = self.f_outputs([x])
        loss_value = outs[0]
        if len(outs[1:]) == 1:
            grad_values = outs[1].flatten().astype('float64')
        else:
            grad_values = np.array(outs[1:]).flatten().astype('float64')
        return loss_value, grad_values



    def get_loss(self, x):
        '''
            This is a helper function to help optimizer to get loss function.

            @x: Input intermediate synthesised image.

            @return: Loss function to be feeded to the optimizer.
        '''
        assert self.loss_value is None
        # Getting loss and grad values.
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return loss_value

    def get_grads(self, x):
        '''
           This is a helper function to help optimizer to get gradient values.

            @x: Input intermediate synthesised image.

            @return: Gradient values to be feeded to the optimizer. 
        '''
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values
        

    
    def buildTexture(self, features='all', iterations=500):
        '''
            This is the main function of this class this wraps everything related to the functionality
            of this project into it.

            @features: VGG19 layers to be selected for texture synthesisation, default being `all` taking 
                        every layer into consideration for synthesisation of the texture.
                        Other options being `pool` taking outputs only from pooling layers or a set of layers
                        named individually
            
            @iterations: Number of times update on x should be carried out default being `500` iterations

            This function does not return anything but saves the synthesised image after every updation.

        '''

        # Creating variables and placeholders
        tex_img = K.variable(self.preprocess_image(self.tex_path))
        gen_img = K.placeholder(shape=self.input_shape)

        # Creating input_tensor by concatenating the two tensors
        input_tensor = K.concatenate([tex_img, gen_img], axis=0)

        # Getting model
        model = VGG19(include_top=False, input_tensor=input_tensor, weights='imagenet')

        # Creating output dictionary for the model.
        outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

        # Initializing loss variable
        loss = K.variable(0.)

        # Creating flag variable for future use
        flag = True

        # Getting names of all the layers in the model
        all_layers = [layer.name for layer in model.layers[1:]]

        # Checking which layers to be used for reference
        if features == 'all':
            feature_layers = all_layers
        elif features == 'pool':
            feature_layers = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool']
        elif isinstance(features, (list, tuple)):
            for f in features:
                if f not in all_layers:
                    flag = False
            if flag:
                feature_layers = features
            else:
                raise ValueError('`features` should be either `all` or `pool` or a set of names from layer.name from model.layers')
        else:
            raise ValueError('`features` should be either `all` or `pool` or a set of names from layer.name from model.layers')


        # Getting features for Texture Image as well as Synthesised Image
        for layer_name in feature_layers:
            layer_features = outputs_dict[layer_name]
            tex_features = layer_features[0, :, :, :]
            gen_features = layer_features[1, :, :, :]
            tex_features = tf.expand_dims(tex_features, axis=0)
            gen_features = tf.expand_dims(gen_features, axis=0)
            
            # Getting loss per layer
            layer_loss = self.get_loss_per_layer(tex_features, gen_features)

            # Calculating total loss
            loss = loss + layer_loss
        
        # Calculating gradient
        grads = tf.gradients(loss, gen_img)
        # Creating a list of loss and gradients 
        outputs = [loss]
        if isinstance(grads, (list, tuple)):
            outputs += grads
        else:
            outputs.append(grads)

        # Using functions features of keras
        self.f_outputs = K.function([gen_img], outputs)

        # Initializing x with base image.    
        x = self.base_img

        # Reducing total loss using fmin_l_bfgs_b function from scipy.
        # For more information regarding fmin_l_bfgs_b refer to https://www.google.co.in
        for i in range(iterations):
            print('Start of iteration', i)
            start_time = time.time()

            # Evalutaing for one iteration.
            x, min_val, info = fmin_l_bfgs_b(func=self.get_loss, x0=x.flatten(), fprime=self.get_grads, maxfun=20)

            print('Current loss value:', min_val)
            # print(info)

            # Deprocessing image
            img = self.deprocess_image(x.copy())
            fname = self.gen_prefix + '_at_iteration_%d.png' % i

            # Saving the synthesised image
            imsave(fname, img)
            end_time = time.time()
            print('Image saved as', fname)
            print('Iteration %d completed in %ds' % (i, end_time - start_time))

    
# Sample run.
tex = DeepTexture('data/inputs/pebbles.jpg')
tex.buildTexture(features='pool')
