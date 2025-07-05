# NEURAL-STYLE-TRANSFER

COMPANY: CODTECH IT SOLLUTIONS

NAME: Venna Leelavathi

INTERN ID: CT06DF1809

DOMAIN: ARTIFICIAL INTELLIGENCE

DURATION: 6 WEEKS

MENTOR: NEELA SANTOSH

DESCRIPTION:

project implements a Neural Style Transfer (NST) model using TensorFlow and the pre-trained VGG19 convolutional neural network. The goal of neural style transfer is to combine the content of one image with the artistic style of another, producing a new image that visually resembles the content image but appears painted or drawn in the style of the second image.

The method is inspired by the work of Gatys et al. (2015), where deep neural networks, particularly convolutional layers, are used to extract high-level features from images. The content features and style features are extracted from different layers of the network. Content is taken from deeper layers (which retain object shapes), while style is taken from shallower layers (which preserve textures and colors).

📌 How the Code Works:
Image Preprocessing:
The script begins by loading and resizing both the content and style images using TensorFlow utilities. The images are processed using VGG19’s expected input format (BGR format with mean subtraction).

Model Setup:
A pre-trained VGG19 network is used, excluding its top classification layers. Specific intermediate layers are selected for extracting style and content features. These layers are chosen because they effectively capture texture (style) and structure (content) at various levels of abstraction.

Feature Extraction:
Using the model, style features are extracted from five convolutional layers, while content features are taken from a single deep layer. For style features, Gram matrices are computed to represent style as spatial correlations between feature maps.

Loss Functions:
The total loss function is a combination of two parts:

Content Loss: Measures the difference between the content features of the generated image and the original content image.

Style Loss: Measures the difference between the style (Gram matrices) of the generated image and the style image.
The loss is weighted to allow control over how much style vs. content is reflected in the output.

Optimization:
An initial image (usually a copy of the content image) is updated using gradient descent to minimize the total loss. This is done iteratively using TensorFlow’s GradientTape and an optimizer like Adam.

Output Generation:
After a fixed number of iterations (e.g., 500–1000), the final image is deprocessed (converted back from VGG19’s format to RGB) and saved as output_stylized_image.jpg.

INPUT:

<img width="253" height="171" alt="Image" src="https://github.com/user-attachments/assets/72cab622-9131-4861-97b9-31a03b2c0c6a" />

<img width="254" height="167" alt="Image" src="https://github.com/user-attachments/assets/3d4510ab-b2e8-4589-9ca7-812285fac669" />


OUTPUT:
<img width="373" height="246" alt="Image" src="https://github.com/user-attachments/assets/03c16d12-a520-45ef-a0bc-9143747a4921" />






