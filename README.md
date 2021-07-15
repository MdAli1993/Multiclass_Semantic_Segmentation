# Geospatial Multiclass Semantic Segmentation
  Implementation of Unet semantic segmentation with pretrained ResNet backbone on geospatial multispectral image data.
## Getting started
### Prerequisites
- Keras = 2.4.3
- Tensorflow = 2.4.1
- Opencv-python = 4.5.1.48
- Numpy = 1.20.1
- GDAL = 3.14
- Matplotlib = 3.3.4
- Sklearn 
- [segmentation_models](https://github.com/qubvel/segmentation_models)

```javascript
pip install segmentation_models
```
### Data preparation
The images and labels are in .tif format, and GDAL library is used for geospatial image data processing.

code to generate ndarray data for training on RGB and multispectral images.

```javascript
mode = "RGB" # multispectral or RGB as mode
# defining start and end channel numbers to stack
def image_mode(RasterCount, mode = mode):
    if mode == "multispectral":
        start_chnl = 1
        end_chnl = RasterCount + 1
        step = 1
        channels = RasterCount
    if mode == "RGB":
        start_chnl = 4
        end_chnl = 1
        step = -1
        channels = 3
    return start_chnl, end_chnl, step, channels

for root, dirs, files in os.walk(image_dir):
    for img_path in glob.glob(os.path.join(root, "*.tif")):
        dataset = gdal.Open(img_path)
        start_chnl, end_chnl, step, channels = image_mode(RasterCount = dataset.RasterCount)
        stacked = np.zeros((dataset.RasterXSize, dataset.RasterYSize, channels), int)
        for x in range(start_chnl, end_chnl, step):
            band = dataset.GetRasterBand(x)
            array = band.ReadAsArray()
            if mode == "RGB":
                stacked[..., -(x-1)] = array #stack bands (4,3,2) for (R,G,B) respectively.
            else:
                stacked[..., x-1] = array
            
        img = np.reshape(stacked,(dataset.RasterXSize, dataset.RasterYSize, channels))
        train_images.append(img)
        train_images_list.append(img_path)    

#Convert list to array for training
train_images = np.array(train_images)
```
Repeat the iteration to generate label list of arrays.

Data splitted into training, testiong, and inference randomly with the approximate ratio of (0.85, 0.1, 0.05) respectively by using sklearn's train_test_split library.
```javascript
from sklearn.model_selection import train_test_split
X1, X_test, y1, y_test = train_test_split(train_images, train_masks_input, test_size = 0.10, random_state = 0)
X_train, X_inference, y_train, y_inference = train_test_split(X1, y1, test_size = 0.05, random_state = 0)
```
Encode the label masks into numerical class vectors, and then convert them into binary matrices.
### Training
Change the shape of input layer in the base model to adapt the N channel
```javascript
if mode == "multispectral":
    base_model = Unet(backbone_name=Backbone, encoder_weights='imagenet', classes = n_classes, activation = 'softmax')

    # map N channels data to 3 channels
    inp = Input(shape=(None, None, N))
    l1 = Conv2D(3, (1, 1))(inp) 
    out = base_model(l1)

    model = Model(inp, out, name=base_model.name)
    
if mode == "RGB":
    model = Unet(backbone_name=Backbone, encoder_weights='imagenet', classes = n_classes, activation = 'softmax')
```
Used Adam as optimizer, categorical_crossentropy for loss calculation, and IoU score for training accuracy metric
```javascript
model.compile('Adam', loss='categorical_crossentropy', metrics= sm.metrics.IOUScore())
```
checkpoint callback to save model based on best validation IoU.
```javascript
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_weights_only=True,
                            monitor='val_iou_score',
                            mode='max',
                            save_best_only=True)
```
Trained the implemented model with batch size 8.

Model is trained for 60 epochs on both RGB and Multispectral data with pretrained weights on imagenet dataset.
|Image Data|Trained no. of Epochs|Train IoU score|Validation Iou score|
|:--------:|:-------------------:|:-------------:|:------------------:|
|Multispectral|60|0.72|0.69|
|RGB|60|0.63|0.62|
### Predictions
predict the trained model on inference images.
```javascript
index = np.random.randint(0, len(X_inference),1) 
pred_img = X_inference[index].reshape((1,64,64,X_inference.shape[3]))
pred_img = normalize(pred_img, axis=1)
y_pred = model.predict(pred_img)
y_pred_argmax = np.argmax(y_pred, axis=3)

prediction = y_pred_argmax.reshape((64,64))
ground_truth = y_inference[index].reshape((64,64))
```
- Predictions on random RGB inference images.

prediction: predicted model output.

Ground truth: label in same pixel values for classes (for better visual comparison).

Label: original label image.
![Capture](https://user-images.githubusercontent.com/58230700/125792388-664c6d2c-1719-42f3-a09f-3ae886b9a73c.PNG)
![RGB](https://user-images.githubusercontent.com/58230700/125793374-ab753eee-04b7-4d92-919e-a65ad5264eb1.PNG)
![zbbPNG](https://user-images.githubusercontent.com/58230700/125793758-982cc699-61f4-4367-9a66-13eef9e575ce.PNG)
- Predictions on random multispectral inference images.
![uhwuif](https://user-images.githubusercontent.com/58230700/125794513-3bf6e6cd-23de-4b1e-9c2b-6986c4b0b326.PNG)
![chdsuifh](https://user-images.githubusercontent.com/58230700/125794543-0259cc39-8a14-4925-9911-8a7e08c1cf62.PNG)
![bshdf](https://user-images.githubusercontent.com/58230700/125794602-0682a2b0-c097-463e-a0db-3bf22123761b.PNG)
