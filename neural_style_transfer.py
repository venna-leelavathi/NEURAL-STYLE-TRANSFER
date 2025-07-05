import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Image loading and preprocessing
def load_and_process_image(path, max_dim=512):
    img = load_img(path)
    img = img_to_array(img)

    # Resize
    scale = max_dim / max(img.shape[:2])
    new_shape = (int(img.shape[1] * scale), int(img.shape[0] * scale))
    img = tf.image.resize(img, new_shape)

    # Batch and preprocess
    img = vgg19.preprocess_input(img)
    return tf.expand_dims(img, axis=0)

def deprocess_image(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = tf.squeeze(x, 0)

    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]  # BGR to RGB
    x = tf.clip_by_value(x, 0, 255).numpy().astype('uint8')
    return x

# Feature extraction
def get_model():
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    style_layers = [
        'block1_conv1', 'block2_conv1',
        'block3_conv1', 'block4_conv1',
        'block5_conv1'
    ]
    content_layer = 'block5_conv2'
    outputs = [vgg.get_layer(name).output for name in style_layers + [content_layer]]
    return Model(inputs=vgg.input, outputs=outputs)

def get_features(model, content_img, style_img):
    style_outputs = model(style_img)
    content_outputs = model(content_img)

    style_features = style_outputs[:-1]
    content_features = content_outputs[-1]
    return style_features, content_features

# Gram Matrix
def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    shape = tf.shape(tensor)
    num_locations = tf.cast(shape[1]*shape[2], tf.float32)
    return result / num_locations

# Loss functions
def compute_loss(model, loss_weights, init_image, gram_style_features, content_feature):
    style_weight, content_weight = loss_weights
    model_outputs = model(init_image)
    style_output_features = model_outputs[:-1]
    content_output_feature = model_outputs[-1]

    style_score = 0
    content_score = 0

    for target, comb in zip(gram_style_features, style_output_features):
        gram_comb = gram_matrix(comb)
        style_score += tf.reduce_mean(tf.square(gram_comb - target))

    content_score = tf.reduce_mean(tf.square(content_output_feature - content_feature))
    total_loss = style_weight * style_score + content_weight * content_score
    return total_loss

# Style Transfer
@tf.function()
def compute_grads(cfg):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**cfg)
    total_loss = all_loss
    return tape.gradient(total_loss, cfg['init_image']), total_loss

def run_style_transfer(content_path, style_path,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False

    content_img = load_and_process_image(content_path)
    style_img = load_and_process_image(style_path)

    style_features, content_feature = get_features(model, content_img, style_img)
    gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]

    init_image = tf.Variable(content_img, dtype=tf.float32)
    opt = tf.optimizers.Adam(learning_rate=5)

    cfg = {
        'model': model,
        'loss_weights': (style_weight, content_weight),
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_feature': content_feature
    }

    best_loss, best_img = float('inf'), None
    for i in range(num_iterations):
        grads, loss = compute_grads(cfg)
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, -127, 127)
        init_image.assign(clipped)

        if loss < best_loss:
            best_loss = loss
            best_img = init_image.numpy()

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss.numpy()}")

    return deprocess_image(best_img)

# Example usage:
if __name__ == "__main__":
    content_path = "content.jpg"
    style_path = "style.jpg"
    output = run_style_transfer(content_path, style_path, num_iterations=500)
    plt.imshow(output)
    plt.axis('off')
    plt.savefig("output_stylized_image.jpg")
    plt.show()