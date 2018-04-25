import os
import io
import hashlib
import yaml
import PIL.Image
import tensorflow as tf
import matplotlib.image as mpimg
from object_detection.utils import dataset_util, label_map_util
from lxml import etree

flags = tf.app.flags
flags.DEFINE_string(
    'data_dir', '', 'Specify root directory to raw dataset and/or to a .yaml file. Seperate multiple datasets with a comma.')
flags.DEFINE_string('annotations_dir', 'Annotations',
                    '(Relative) path to annotations directory. (Needed for XML!)')
flags.DEFINE_string('output_path', '',
                    'Path to output TFRecord e.g.: data/train.record')
flags.DEFINE_string('label_map_path', '',
                    'Path to label map proto e.g.: data/label_map.pbtxt')
flags.DEFINE_boolean('ignore_difficult_instances', False,
                     'Whether to ignore difficult instances')
FLAGS = flags.FLAGS


def load_image(path):
    return mpimg.imread(path)


def png_to_jpeg(png_file, jpeg_file):
    """
    Convert PNG images to JPEG format
    :param png_file: full path of .png file
    :param jpeg_file: full path of .jpeg file
    """
    im = PIL.Image.open(png_file)
    rgb_im = im.convert('RGB')
    rgb_im.save(jpeg_file, 'JPEG')


def create_jpg_imgs(image_path):
    # create jpeg subdirectory
    jpeg_dir = os.path.join(os.path.dirname(image_path), 'jpeg')
    if not os.path.exists(jpeg_dir):
        os.makedirs(jpeg_dir)

    if image_path.split('.')[-1].lower() == 'png':
        png_file = image_path
        image_path = image_path.replace(
            os.path.dirname(image_path), jpeg_dir)
        image_path = image_path.replace('.png', '.jpg')
        # convert images to jpeg if they don't already exist
        if not os.path.isfile(image_path):
            png_to_jpeg(png_file, image_path)
    return image_path


def get_imgs_from_yaml(input_yaml, riib=False):
    """ Gets all labels within label file
    Note that RGB images are 1280x720 and RIIB images are 1280x736.
    :param input_yaml: Path to yaml file
    :param riib: If True, change path to labeled pictures
    :return: images: Labels for traffic lights
    """
    images = yaml.load(open(input_yaml, 'rb').read())

    width, height = None, None
    for image in images:
        image['path'] = os.path.abspath(os.path.join(
            os.path.dirname(input_yaml), image['path']))
        if width is None and height is None:
            ## assume all images have the same properties
            img = load_image(image['path'])
            height = img.shape[0]
            width = img.shape[1]

        image.update({'height': height, 'width': width})
        if riib:
            image['path'] = image['path'].replace('.png', '.pgm')
            image['path'] = image['path'].replace('rgb/train', 'riib/train')
            image['path'] = image['path'].replace('rgb/test', 'riib/test')
            for box in image['boxes']:
                box['y_max'] = box['y_max'] + 8
                box['y_min'] = box['y_min'] + 8

        image['path'] = create_jpg_imgs(image['path'])

    return images


def create_tf_record(data, label_map_dict, is_yaml=False, ignore_difficult_instances=False):
    """
    Convert XML derived dict to tf.Example proto.

    Notice that this function normalizes the bounding box coordinates provided
    by the raw data.

    Args:
    :param data: dict holding (XML or YAML) fields for a single image (obtained by running dataset_util.recursive_parse_xml_to_dict)
    :param label_map_dict: A map from string label names to integers ids.
    :param ignore_difficult_instances: Whether to skip difficult instances in the dataset  (default: False).

    Returns:
    :return tf_example: The converted tf.Example.

    Raises:
    ValueError: if the image pointed to by data['filename'] is not a valid JPEG
    """

    with tf.gfile.GFile(data['path'], 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    if image.format != 'JPEG':
        raise ValueError('Image format not JPG')
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    ymin = []
    xmax = []
    ymax = []
    classes = []
    classes_text = []
    truncated = []
    poses = []
    difficult_obj = []

    if is_yaml:
        width = int(data['width'])
        height = int(data['height'])
        filename = data['path'].encode('utf8')
        for box in data['boxes']:
            difficult_obj.append(0)

            xmin.append(float(box['x_min']) / width)
            ymin.append(float(box['y_min']) / height)
            xmax.append(float(box['x_max']) / width)
            ymax.append(float(box['y_max']) / height)
            classes_text.append(box['label'].encode('utf8'))
            classes.append(label_map_dict[box['label']])
            truncated.append(0)
            poses.append(r'Unspecified'.encode('utf8'))
    else:
        width = int(data['size']['width'])
        height = int(data['size']['height'])
        filename = data['filename'].encode('utf8')

        for obj in data['object']:
            difficult = bool(int(obj['difficult']))
            if ignore_difficult_instances and difficult:
                continue

            difficult_obj.append(int(difficult))

            xmin.append(float(obj['bndbox']['xmin']) / width)
            ymin.append(float(obj['bndbox']['ymin']) / height)
            xmax.append(float(obj['bndbox']['xmax']) / width)
            ymax.append(float(obj['bndbox']['ymax']) / height)
            classes_text.append(obj['name'].encode('utf8'))
            classes.append(label_map_dict[obj['name']])
            truncated.append(int(obj['truncated']))
            poses.append(obj['pose'].encode('utf8'))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(r'jpg'.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
        'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
        'image/object/truncated': dataset_util.int64_list_feature(truncated),
        'image/object/view': dataset_util.bytes_list_feature(poses),
    }))
    return tf_example


def main(_):

    label_map_dict = label_map_util.get_label_map_dict(
        FLAGS.label_map_path)  # label map --> FLAGS.label_map
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

    dataset_list = FLAGS.data_dir.split(',')
    for dataset in dataset_list:
        if dataset.split('.')[-1] == r'yaml':
            ## FOR YAML
            examples_list = get_imgs_from_yaml(dataset)
            for example in examples_list:
                tf_example = create_tf_record(
                    example, label_map_dict, is_yaml=True, ignore_difficult_instances=FLAGS.ignore_difficult_instances)
                writer.write(tf_example.SerializeToString())
        else:
            ## FOR XML
            annotations_dir = os.path.join(dataset, FLAGS.annotations_dir)
            examples_list = [os.path.splitext(name)[0] for name in os.listdir(
                dataset) if os.path.isfile(os.path.join(dataset, name))]
            for example in examples_list:
                path = os.path.join(annotations_dir, example + '.xml')
                with tf.gfile.GFile(path, 'r') as fid:
                    xml_str = fid.read()
                xml = etree.fromstring(xml_str)
                data = dataset_util.recursive_parse_xml_to_dict(xml)[
                    'annotation']
                # convert the path to the current file directory
                data['path'] = os.path.join(os.path.abspath(
                    dataset), os.path.basename(data['path']))

                data['path'] = create_jpg_imgs(data['path'])

                tf_example = create_tf_record(
                    data, label_map_dict, ignore_difficult_instances=FLAGS.ignore_difficult_instances)
                writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
  tf.app.run()
