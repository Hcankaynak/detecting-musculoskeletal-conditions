import matplotlib.image as mpimg
from keras import backend as K
%matplotlib inline
K.clear_session()
from keras.applications.densenet import preprocess_input
import pandas as pd
from keras.applications.densenet import decode_predictions
from keras.preprocessing import image
import cv2

model = load_model("drive/My Drive/CNN/DenseNet169.h5")
Y_pred = model.predict_generator(test_set, steps = len(test_set))
y_pred = np.argmax(Y_pred, axis=1)
print('Classification Report')
target_names = ['negative', 'positive']
print(classification_report(test_set.classes, y_pred, target_names=target_names))

img_path = 'test_set_wrist/val_0/342.png'
img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.0
plt.imshow(img_tensor[0])
plt.show()
print(img_tensor.shape)

#x = image.img_to_array(img_te)
#x = np.expand_dims(x, axis=0)
#img_tensor = preprocess_input(img_tensor)
preds = model.predict(img_tensor)
argmax = np.argmax(preds)

output = model.output[:, argmax]
last_conv_layer = model.get_layer('conv5_block32_2_conv')
grads = K.gradients(output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([img_tensor])

for i in range(32):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

img = cv2.imread(img_path)

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
hif = .8
superimposed_img = heatmap * hif + img

output = 'drive/My Drive/CNN/heatmap_1.png'
cv2.imwrite(output, superimposed_img)

img=mpimg.imread(output)

plt.imshow(img)
plt.axis('off')