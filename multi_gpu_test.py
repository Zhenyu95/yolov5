import tensorflow as tf
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
import tensorflow_datasets as tfds
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

DATA_DIR = '/Users/zhenyu/Desktop/data/image/'
IMG_HEIGHT = 4032
IMG_WIDTH = 3008
IMG_DIR = 'image'
LABEL_DIR = 'binary'
AUTOTUNE = tf.data.AUTOTUNE


# specify that we want to use Mirrored Strategy
strategy = tf.distribute.MirroredStrategy()
# Check the number of Decives found on the machine
print('*****'*10,'Number of devices: {}'.format(strategy.num_replicas_in_sync), '*****'*10)


def get_dataset(data_dir=DATA_DIR, val_ratio=0.2):
    # get the number of instances
    image_count = len([name for name in os.listdir(data_dir) if (os.path.isfile(os.path.join(data_dir, name)) and str(name).endswith('.jpeg'))])
    # create a Dataset with all the jpg files in the directory
    list_ds = tf.data.Dataset.list_files(str(data_dir+'*.jpeg'), shuffle=False)
    # shuffle the Dataset
    list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration=False)
    # split train & val
    val_size = int(image_count * val_ratio)
    train_ds = list_ds.skip(val_size)
    val_ds = list_ds.take(val_size)
    return train_ds, val_ds


def get_img(file_path, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    # load the file
    img = tf.io.read_file(file_path)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # img = tf.image.convert_image_dtype(img, tf.uint8)
    # resize the image to the desired size
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def get_label(file_name, img_height=IMG_HEIGHT, img_width=IMG_WIDTH):
    # replace "image" from the filename with "binary", change the directory
    binary_path = tf.strings.regex_replace(file_name, 'image', 'binary')
    label = tf.io.read_file(binary_path)
    label = tf.io.decode_jpeg(label, channels=0)
    # label = tf.image.convert_image_dtype(label, tf.uint8)
    label = tf.image.resize(label, [img_height, img_width])
    return label


def process_path(file_path):
    img = get_img(file_path)
    label = get_label(file_path)
    return img, label

train_ds, val_ds = get_dataset()

train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)


# Batch the input data
BUFFER_SIZE = len(train_ds)
BATCH_SIZE_PER_REPLICA = 8
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=2)
    ds = ds.batch(2)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


# Create Distributed Datasets from the datasets
train_ds = train_ds.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
train_dist_dataset = strategy.experimental_distribute_dataset(train_ds)
val_dist_dataset = strategy.experimental_distribute_dataset(val_ds)

# Create the model architecture
def create_model(img_size=(4032, 3008, 3)):
    inputs = Input(img_size)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    return model


with strategy.scope():
    # We will use sparse categorical crossentropy as always. But, instead of having the loss function
    # manage the map reduce across GPUs for us, we'll do it ourselves with a simple algorithm.
    # Remember -- the map reduce is how the losses get aggregated
    # Set reduction to `none` so we can do the reduction afterwards and divide byglobal batch size.
    # Otherwise, the loss from different devices will be aggregated automatically
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(masks, predictions):
        # Compute Loss uses the loss object to compute the loss
        # Notice that per_example_loss will have an entry per GPU
        # so in this case there'll be 2 -- i.e. the loss for each replica
        per_example_loss = loss_object(masks, predictions)
        # You can print it to see it -- you'll get output like this:
        # Tensor("sparse_categorical_crossentropy/weighted_loss/Mul:0", shape=(48,), dtype=float32, device=/job:localhost/replica:0/task:0/device:GPU:0)
        # Tensor("replica_1/sparse_categorical_crossentropy/weighted_loss/Mul:0", shape=(48,), dtype=float32, device=/job:localhost/replica:0/task:0/device:GPU:1)
        # Note in particular that replica_0 isn't named in the weighted_loss -- the first is unnamed, the second is replica_1 etc
        print(per_example_loss)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

    # We'll just reduce by getting the average of the losses
    val_loss = tf.keras.metrics.Mean(name='val_loss')

    # Accuracy on train and test will be SparseCategoricalAccuracy
    train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
    val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')

    # Optimizer will be Adam
    optimizer = tf.keras.optimizers.Adam()

    # Create the model within the scope
    model = create_model()

    # `run` replicates the provided computation and runs it
# with the distributed input.
@tf.function
def distributed_train_step(dataset_inputs):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
    #tf.print(per_replica_losses.values)
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

def train_step(inputs):
    images, masks = inputs
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(masks, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(masks, predictions)
    return loss

#######################
# Test Steps Functions
#######################
@tf.function
def distributed_test_step(dataset_inputs):
    return strategy.run(test_step, args=(dataset_inputs,))

def test_step(inputs):
    images, masks = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(masks, predictions)

    val_loss.update_state(t_loss)
    val_accuracy.update_state(masks, predictions)


EPOCHS = 10
for epoch in range(EPOCHS):
    # Do Training
    total_loss = 0.0
    num_batches = 0
    for batch in train_dist_dataset:
        total_loss += distributed_train_step(batch)
        num_batches += 1
    train_loss = total_loss / num_batches

    # Do Testing
    for batch in val_dist_dataset:
        distributed_test_step(batch)

    template = ("Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, " "Test Accuracy: {}")

    print (template.format(epoch+1, train_loss, train_accuracy.result()*100, val_loss.result(), val_accuracy.result()*100))

    val_loss.reset_states()
    train_accuracy.reset_states()
    val_accuracy.reset_states()