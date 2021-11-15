
import coremltools as ct
from PIL import Image
import time

def export_model():
    import tensorflow as tf
    import urllib
    keras_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224,224,3), alpha=1.0, minimalistic=False, include_top=True,
        weights='imagenet', input_tensor=None, classes=1000, pooling=None,
        dropout_rate=0.2, classifier_activation='softmax',
        include_preprocessing=True
    )


    label_url = 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt'
    class_labels = urllib.request.urlopen(label_url).read().splitlines()
    class_labels = class_labels[1:] # remove the first class which is background
    assert len(class_labels) == 1000

    # make sure entries of class_labels are strings
    for i, label in enumerate(class_labels):
        if isinstance(label, bytes):
            class_labels[i] = label.decode("utf8")
            
    image_input = ct.ImageType(shape=(1, 224, 224, 3,),
                            bias=[-1,-1,-1], scale=1/127)

    classifier_config = ct.ClassifierConfig(class_labels)

    model = ct.convert(
        keras_model, inputs=[image_input], classifier_config=classifier_config,
    )

    model.input_description["input_1"] = "Input image to be classified"
    model.output_description["classLabel"] = "Most likely image category"

    tf.saved_model.save(model, '/tf2coreml/')


def pred():
    example_image = Image.open("/Users/zhenyu/Desktop/bird.jpg").resize((224, 224))
    start_time = time.time()
    out_dict = model.predict({"input_1": example_image})
    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == '__main__':
    export_model()
    pred()
    