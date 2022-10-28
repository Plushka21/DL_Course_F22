# # Deep Learning
# ## Assignment 1
# ---
# ## Anton Buguev BS19-RO-01
# #### a.buguev@innopolis.university
# #### 20 September 2022

import os
import sys
import scipy.io
import imageio
import scipy.misc
import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import import_ipynb

# since most of the code is written for tensorflow-1 which is not supperted anymore
# we need to import tensorflow-2 as tensorflow-1 to allow all functions
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# %matplotlib inline

print()
print("Starting...")
print()

content_image_name = sys.argv[1]
style_image_name = sys.argv[2]
generated_name = sys.argv[3]

gif_folder = "output"
if not os.path.isdir(gif_folder):
    os.mkdir(gif_folder)

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

content_image = imageio.imread(content_image_name)
imageio.imwrite(gif_folder + "/-1.png", content_image)
# imshow(content_image)


def compute_content_cost(a_C, a_G):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = tf.reshape(a_C, [n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, [n_H * n_W, n_C])
    
    # compute the cost with tensorflow (≈1 line)
    J_content = 1./(4 * n_H * n_W * n_C)*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    
    ### END CODE HERE ###
    
    return J_content

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_content = compute_content_cost(a_C, a_G)
    # print("J_content = " + str(J_content.eval()))

style_image = imageio.imread(style_image_name)
# imshow(style_image)


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    ### START CODE HERE ### (≈1 line)
    GA = tf.matmul(A, tf.matrix_transpose(A))
    
    ### END CODE HERE ###
    
    return GA

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2*1], mean=1, stddev=4)
    GA = gram_matrix(A)
    
    # print("GA = " + str(GA.eval()))


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    ### START CODE HERE ###
    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = tf.reshape(tf.transpose(a_S, perm=[3, 2, 1, 0]), [n_C, n_H * n_W])
    a_G = tf.reshape(tf.transpose(a_G, perm=[3, 2, 1, 0]), [n_C, n_H * n_W])

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG))) / (4 * n_C ** 2 * (n_H * n_W) ** 2)
    
    ### END CODE HERE ###
    
    return J_style_layer

tf.reset_default_graph()

with tf.Session() as test:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random_normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)
    
    # print("J_style_layer = " + str(J_style_layer.eval()))

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- our tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Select the output tensor of the currently selected layer
        out = model[layer_name]

        # Set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    
    ### START CODE HERE ### (≈1 line)
    J = alpha*J_content + beta*J_style
    
    ### END CODE HERE ###
    
    return J

tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(3)
    J_content = np.random.randn()    
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    # print("J = " + str(J))


tf.reset_default_graph()

# Start interactive session
sess = tf.InteractiveSession()

content_image = content_image[:,:,:3] #For some png images. Can be removed!
content_image = reshape_and_normalize_image(content_image)


style_image = style_image[:,:,:3] #For some png images. Can be removed!
style_image = reshape_and_normalize_image(style_image)


generated_image = generate_noise_image(content_image)
# imshow(generated_image[0])

model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")


# 1. Assign the content image to be the input to the VGG model.
# 2. Set a_C to be the tensor giving the hidden layer activation for layer "conv4_2".
# 3. Set a_G to be the tensor giving the hidden layer activation for the same layer. 
# 4. Compute the content cost using a_C and a_G.

# Assign the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Select the output tensor of layer conv4_2
out = model['conv4_2']

# Set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# Set a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Compute the content cost
J_content = compute_content_cost(a_C, a_G)

# Assign the input of the model to be the "style" image 
sess.run(model['input'].assign(style_image))

# Compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

### START CODE HERE ### (1 line)
J = total_cost(J_style, J_content, alpha=10, beta=40)
### END CODE HERE ###

# define optimizer (1 line)
optimizer = tf.train.AdamOptimizer(2.0)
# define train_step (1 line)
train_step = optimizer.minimize(J)

def model_nn(sess, input_image, num_iterations=200):
    # Initialize global variables (you need to run the session on the initializer)
    ### START CODE HERE ### (1 line)
    sess.run(tf.global_variables_initializer())
    ### END CODE HERE ###

    # Run the noisy input image (initial generated image) through the model. Use assign().
    ### START CODE HERE ### (1 line)
    sess.run(model['input'].assign(input_image))
    ### END CODE HERE ###

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        ### START CODE HERE ### (1 line)
        sess.run(train_step)
        ### END CODE HERE ###

        # Compute the generated image by running the session on the current model['input']
        ### START CODE HERE ### (1 line)
        generated_image = sess.run(model['input'])
        ### END CODE HERE ###

        # Print every 20 iteration.
        if i % 10 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            print()

            # save current generated image in the "/output" directory
            # generated_image[0] = imageio.astype(np.uint8)
            save_image(gif_folder + "/" + str(i) + ".png", generated_image)

    # save last generated image
    save_image(generated_name, generated_image)
    save_image(gif_folder + "/" + str(num_iterations) + ".png", generated_image)

    return generated_image

print()
print("Image generation started:")
print()
model_nn(sess, generated_image)
print()
print("Image generation complete!")

# Function to generate GIF
def make_gif(frame_folder):
    image_numbers = sorted(int(name[:-4]) for name in os.listdir(frame_folder))
    frames = [Image.open(frame_folder + "/" + str(image) + ".png") for image in image_numbers]
    frame_one = frames[0]
    frame_one.save("process.gif", format="GIF", append_images=frames,
               save_all=True, duration=500, loop=0)
print()
print("GIF generation started...")
make_gif(gif_folder)
print("GIF generation complete!")
print()
print("The whole code executed!")