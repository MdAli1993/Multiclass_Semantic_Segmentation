# Geospatial Semantic Segmentation
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
### Data Preparation
The .tif image format data is used for this project.

code to generate ndarray data for training on RGB and multispectral images.

```javascript
# path to images and labels.
image_dir = "path to image folder"
label_dir = "path to label folder"

# training image info as a list
train_images = []
train_images_list = []

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
                stacked[..., -(x-1)] = array
            else:
                stacked[..., x-1] = array
            
        img = np.reshape(stacked,(dataset.RasterXSize, dataset.RasterYSize, channels))
        train_images.append(img)
        train_images_list.append(img_path)    

#Convert list to array for training
train_images = np.array(train_images)
```
Repeat the iteration to generate label list of arrays.

Encode the label masks into numerical class vectors, and then convert them into binary matrices.
### Training
```javascript
# using ResNet as model backbone with imagenet pretrained weights
Backbone = 'resnet34'
preprocess_input1 = sm.get_preprocessing(Backbone)

X_train1 = preprocess_input1(X_train)
X_test1 = preprocess_input1(X_test)

N = X_train.shape[3] # no. of channels

if mode == "multispectral":
    base_model = Unet(backbone_name=Backbone, encoder_weights='imagenet', classes = n_classes, activation = 'softmax')

    # map N channels data to 3 channels
    inp = Input(shape=(None, None, N))
    l1 = Conv2D(3, (1, 1))(inp) 
    out = base_model(l1)

    model = Model(inp, out, name=base_model.name)
    
if mode == "RGB":
    model = Unet(backbone_name=Backbone, encoder_weights='imagenet', classes = n_classes, activation = 'softmax')

#start training with previously trained model weights.
model.load_weights('path to load pre-trained weights') 

model.compile('Adam', loss='categorical_crossentropy', metrics= sm.metrics.IOUScore())

model.summary()

# filepath to save training model checkpoins
checkpoint_filepath = "path to save model weights"

# checkpoint callback to save model based on best validation IoU.
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                            filepath=checkpoint_filepath,
                            save_weights_only=True,
                            monitor='val_iou_score',
                            mode='max',
                            save_best_only=True)

model.fit(X_train1,
           y_train_cat,
           batch_size = 8,
           epochs = 60,
           verbose = 1,
           validation_data = (X_test1, y_test_cat),
           callbacks=[model_checkpoint_callback])
```
### Visualization
```javascript
# prediction on random inference images
index = np.random.randint(0, len(X_inference),1) 
pred_img = X_inference[index].reshape((1,64,64,X_inference.shape[3]))
pred_img = normalize(pred_img, axis=1)
y_pred = model.predict(pred_img)
y_pred_argmax = np.argmax(y_pred, axis=3)

prediction = y_pred_argmax.reshape((64,64))
ground_truth = y_inference[index].reshape((64,64))

# splitting the data to get a same inference label image.
X1_list, X_test_list, y1_list, y_test_list = train_test_split(train_images_list, train_masks_list, test_size = 0.10, random_state = 0)
X_train_list, X_inference_list, y_train_list, y_inference_list = train_test_split(X1_list, y1_list, test_size = 0.05, random_state = 0)
inference_image_name = np.array(y_inference_list)[index][0]
inference_image = cv2.imread(inference_image_name)

# visualization on inference images.
print('\033[1m' +"\t\t\t\t\tPredictions on "+mode+" mode")
fig, axes = plt.subplots(nrows=1,ncols=3,figsize=(12,12))
plt.sca(axes[0]); 
plt.imshow(prediction,cmap="viridis"); plt.title('prediction')
plt.axis("off")
plt.sca(axes[1]); 
plt.imshow(ground_truth,cmap="viridis"); plt.title('Ground truth')
plt.axis("off")
plt.sca(axes[2]); 
plt.imshow(inference_image); plt.title('Label')
plt.axis("off")
plt.tight_layout()
plt.show()
```
